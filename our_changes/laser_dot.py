

"""
laser_dot.py — Simulated laser pointer for the Stretch MuJoCo digital twin.

HOW IT WORKS
------------
A physical laser pointer leaves a red dot wherever its beam hits a surface.
The robot sees the dot through its camera — that's all it needs.

In simulation we replicate this by letting the user RIGHT-CLICK directly on
the OpenCV camera window. The clicked pixel already IS the laser dot position
in camera space. We then use the depth frame to lift that pixel into 3D world
coordinates, exactly as the real pipeline would with a RealSense.

Additionally, update_from_frame() detects a REAL physical red laser pointer
automatically by color (high saturation + high brightness red in HSV) and
small blob size, without requiring any clicks.

WHY RIGHT-CLICK ON THE CAMERA WINDOW?
--------------------------------------
- Left-click is reserved for future UI use.
- The MuJoCo server runs in a separate process — we cannot hook its GLFW
  window from here. The OpenCV window lives in our process, so callbacks
  work perfectly.
- Clicking on what you SEE in the camera is actually more natural than
  clicking on the 3D viewport — it's closer to how a real user would
  interact (point laser at object, robot sees the dot).

USAGE
-----
    from laser_dot import LaserDotSimulator
    from stretch_toolkit import HEAD_CAMERA, HEAD_RGB_CAMERA

    laser = LaserDotSimulator(HEAD_CAMERA, window_name="Head Camera")

    # In your loop, BEFORE cv2.imshow:
    rgb   = HEAD_RGB_CAMERA.get_frame()
    depth = HEAD_DEPTH_CAMERA.get_frame()

    # Option A — automatic physical laser detection:
    laser.update_from_frame(rgb, depth)

    # Option B — manual right-click (original behaviour):
    display = laser.overlay(rgb)
    cv2.imshow("Head Camera", display)
    cv2.waitKey(1)

    # Read results:
    pixel = laser.get_dot_pixel()         # (u, v) or None
    world = laser.get_dot_world()         # (x, y, z) metres or None
    if laser.dot_updated():
        print("New target:", world)
"""

import threading
import numpy as np
import cv2

# Dot appearance
_DOT_COLOR_BGR = (0, 0, 255)   # bright red
_DOT_RADIUS    = 8             # pixels
_DOT_OUTLINE   = (0, 0, 120)   # dark red ring for contrast


class LaserDotSimulator:
    """
    Captures right-clicks on an OpenCV camera window and treats them as
    laser pointer selections. Uses the depth frame to compute the 3D world
    position of the selected point.

    Also supports automatic detection of a physical red laser pointer via
    update_from_frame(rgb_frame, depth_frame).

    Parameters
    ----------
    head_camera : DepthCamInfo
        The HEAD_CAMERA object from stretch_toolkit (has .get_depth() and
        intrinsics for both RGB and depth).
    window_name : str
        The cv2.imshow() window name to attach the mouse callback to.
        Must match exactly what you pass to cv2.imshow().
    """

    def __init__(self, head_camera, window_name: str = "Head Camera"):
        self._camera   = head_camera
        self._win_name = window_name

        self._lock        = threading.Lock()
        self._dot_pixel   = None   # (u, v) in RGB camera frame
        self._dot_world   = None   # (x, y, z) in metres
        self._dot_updated = False  # True for one read after a new click

        self._callback_registered = False
        self._scale_x = 2.0  # default: frame shown at 2x (848x480 from 424x240)
        self._scale_y = 2.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dot_pixel(self):
        """Return the (u, v) pixel of the dot in the RGB frame, or None."""
        with self._lock:
            return self._dot_pixel

    def get_dot_world(self):
        """Return the (x, y, z) world coordinate in metres, or None."""
        with self._lock:
            return self._dot_world

    def dot_updated(self) -> bool:
        """True exactly once after each new right-click or laser detection. Call once per loop."""
        with self._lock:
            if self._dot_updated:
                self._dot_updated = False
                return True
            return False

    def clear(self):
        """Clear the current dot (call after robot picks up the object)."""
        with self._lock:
            self._dot_pixel   = None
            self._dot_world   = None
            self._dot_updated = False
        print("[LaserDot] Dot cleared.")

    def register_callback(self):
        """
        Attach the right-click handler to the OpenCV window.

        Must be called AFTER the first cv2.imshow() for the window,
        because OpenCV only creates the window on the first imshow call.
        Call this once in your setup, then it stays registered.
        """
        cv2.setMouseCallback(self._win_name, self._on_mouse)
        self._callback_registered = True
        print(f"[LaserDot] Right-click on '{self._win_name}' to place the laser dot.")

    def overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw the laser dot onto a camera frame copy.

        Pass the raw RGB frame; get back a copy with the red dot drawn.
        Returns the original frame unchanged if no dot is active or frame is None.
        """
        if frame is None:
            return frame

        with self._lock:
            pixel = self._dot_pixel

        if pixel is None:
            return frame

        out = frame.copy()
        u, v = pixel
        cv2.circle(out, (u, v), _DOT_RADIUS + 2, _DOT_OUTLINE, 2)       # dark ring
        cv2.circle(out, (u, v), _DOT_RADIUS, _DOT_COLOR_BGR, -1)         # red fill
        cv2.circle(out, (u - 2, v - 2), 2, (255, 255, 255), -1)          # specular
        return out

    # ------------------------------------------------------------------
    # Automatic physical laser pointer detection
    # ------------------------------------------------------------------

    def update_from_frame(self, rgb_frame: np.ndarray, depth_frame) -> bool:
        """
        Automatically detect a physical red laser pointer dot in rgb_frame
        and update dot_pixel / dot_world if found.

        The laser pointer produces a very small, highly saturated, very bright
        red blob — different from ordinary red objects (like the red cylinder)
        which are larger and less saturated.

        Detection thresholds (tune to your camera and laser):
          - Hue:        [0..10] ∪ [160..180]   (red wraps in HSV)
          - Saturation: ≥ 150   (laser is deeply saturated)
          - Value:      ≥ 200   (laser overexposes the sensor)
          - Blob area:  10..300 px²  (tiny — ignore large red objects)

        Parameters
        ----------
        rgb_frame : np.ndarray
            BGR frame from HEAD_RGB_CAMERA or WRIST_RGB_CAMERA.
        depth_frame :
            Depth frame from HEAD_DEPTH_CAMERA (same timestamp preferred).

        Returns
        -------
        bool
            True if a laser dot was detected and the internal state updated.
        """
        if rgb_frame is None or depth_frame is None:
            return False

        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)

        # Red laser: high saturation + very high brightness
        # Tweak S_MIN / V_MIN if your laser appears dim or causes false positives
        S_MIN = 150
        V_MIN = 200
        AREA_MAX = 300   # px² — raise if the laser dot looks large in camera
        AREA_MIN = 10    # px² — ignore single-pixel noise

        mask1 = cv2.inRange(hsv, np.array([0,   S_MIN, V_MIN]),
                                  np.array([10,  255,   255  ]))
        mask2 = cv2.inRange(hsv, np.array([160, S_MIN, V_MIN]),
                                  np.array([180, 255,   255  ]))
        mask = mask1 | mask2

        # Gaussian blur before threshold: laser overexposes the sensor and
        # often creates a bright white centre surrounded by a red ring.
        # Blurring merges them into one solid blob.
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False

        # Keep only small blobs (laser dot is tiny)
        candidates = [c for c in contours
                      if AREA_MIN < cv2.contourArea(c) < AREA_MAX]
        if not candidates:
            return False

        # Among candidates pick the brightest blob in the V channel
        # (most likely to be the laser rather than a red surface highlight)
        def _brightness(contour):
            tmp = np.zeros(hsv.shape[:2], dtype=np.uint8)
            cv2.drawContours(tmp, [contour], -1, 255, -1)
            return cv2.mean(hsv[:, :, 2], mask=tmp)[0]

        best = max(candidates, key=_brightness)
        M = cv2.moments(best)
        if M['m00'] == 0:
            return False

        u = int(M['m10'] / M['m00'])
        v_coord = int(M['m01'] / M['m00'])

        # Reuse select_pixel which handles depth lookup + 3D unprojection
        self.select_pixel(u, v_coord, depth_frame)
        return True

    # ------------------------------------------------------------------
    # Shared pixel → 3D logic (used by both click and auto-detect)
    # ------------------------------------------------------------------

    def select_pixel(self, u: int, v: int, depth_frame):
        """
        Select a pixel in NATIVE frame coordinates (not scaled display coords).
        e.g. for a 424x240 frame, center is (212, 120).

        Computes 3D camera-frame coordinates from depth and camera intrinsics,
        then stores the result in dot_pixel and dot_world.
        """
        depth = self._camera.get_depth((u, v), depth_image=depth_frame)
        if depth is None or depth <= 0:
            print(f"[LaserDot] No valid depth at pixel ({u}, {v}). Try again.")
            return

        fx = self._camera.fx
        fy = self._camera.fy
        cx = self._camera.cx
        cy = self._camera.cy

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        with self._lock:
            self._dot_pixel   = (u, v)
            self._dot_world   = (float(x_cam), float(y_cam), float(z_cam))
            self._dot_updated = True

        print(f"[LaserDot] ✓ Dot at pixel ({u}, {v}) → "
              f"camera-frame ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) m")

    def set_display_scale(self, scale_x: float, scale_y: float):
        """
        Call this if the display window is scaled relative to the raw frame.
        e.g. if 424x240 frame is shown at 848x480, call set_display_scale(2.0, 2.0).
        Click coordinates will be mapped back to real frame coordinates.
        """
        self._scale_x = scale_x
        self._scale_y = scale_y

    # ------------------------------------------------------------------
    # Internal: OpenCV mouse callback
    # ------------------------------------------------------------------

    def _on_mouse(self, event, u, v, flags, param):
        """OpenCV mouse callback — fires on right-click in the camera window."""
        if event != cv2.EVENT_RBUTTONDOWN:
            return

        # Map scaled display coordinates back to real frame coordinates
        u = int(u / self._scale_x)
        v = int(v / self._scale_y)

        # Grab fresh depth frame — the main loop must poll depth every frame
        # to keep the watchdog from deregistering the camera between clicks.
        depth_frame = self._camera.depth_cam.get_frame()
        if depth_frame is None:
            print("[LaserDot] Depth camera not ready — ensure HEAD_DEPTH_CAMERA.get_frame() ")
            print("          is called every loop iteration to keep the watchdog alive.")
            return

        depth = self._camera.get_depth((u, v), depth_image=depth_frame)

        if depth is None or depth <= 0:
            print(f"[LaserDot] No valid depth at pixel ({u}, {v}). "
                  "Try clicking directly on a solid object.")
            return

        # Unproject RGB pixel + depth → 3D camera-frame point
        fx = self._camera.fx
        fy = self._camera.fy
        cx = self._camera.cx
        cy = self._camera.cy

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        world = (float(x_cam), float(y_cam), float(z_cam))

        with self._lock:
            self._dot_pixel   = (u, v)
            self._dot_world   = world
            self._dot_updated = True

        print(f"[LaserDot] ✓ Dot at pixel ({u}, {v}) → "
              f"camera-frame ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f}) m")


