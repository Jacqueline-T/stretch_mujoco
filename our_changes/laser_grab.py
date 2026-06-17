"""
laser_grab.py – Aim camera, select object, robot picks it up.

CONTROLS (in Select Target window)
-----------------------------------
  I / K   Tilt camera up / down
  J / L   Pan camera left / right
  B       Select blue box
  R       Select red cylinder
  Q       Quit
"""

from stretch_toolkit import (
    controller, teleop, merge_proportional, locate_object,
    BACKEND_NAME, HEAD_CAMERA, HEAD_RGB_CAMERA,
    WRIST_CAMERA, WRIST_RGB_CAMERA, StateController
)
from stretch_toolkit import _sim  
from stretch_toolkit.robot_transforms import RobotTransforms
from laser_dot import LaserDotSimulator
from laser_seek import LaserSeeker
import time
import math
import cv2
import numpy as np

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

HEAD_VEL        = 0.6
TRACKER_TIMEOUT = 10
DISP_W, DISP_H  = 848, 480   # display size (2x native 424x240)


def find_object_by_color(rgb_frame, target='blue'):
    """
    Returns (centroid, bbox) or (None, None).
    Blue: strict saturation + value to exclude sky (light, desaturated).
    Red: wide hue range both ends.
    """
    if rgb_frame is None:
        return None, None

    hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2HSV)

    if target == 'blue':
        # Sky is hue ~100-110 but LOW saturation (~30-80) and HIGH value.
        # The box is HIGHLY saturated (>150) and mid-dark value (>60).
        # Requiring high saturation excludes sky completely.
        mask = cv2.inRange(hsv,
                           np.array([105, 150, 40]),   # hue, sat_min=150, val_min=40
                           np.array([130, 255, 200]))  # val_max=200 excludes bright sky
    else:  # red
        mask1 = cv2.inRange(hsv, np.array([0,   60, 40]),  np.array([15,  255, 255]))
        mask2 = cv2.inRange(hsv, np.array([155, 60, 40]),  np.array([180, 255, 255]))
        mask = mask1 | mask2

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 300:
            x, y, w, h = cv2.boundingRect(largest)
            M = cv2.moments(largest)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                return (cx, cy), (x, y, w, h)
    return None, None


class RobustTracker:
    def __init__(self, target_color):
        self.target_color  = target_color
        self.tracker       = None
        self.last_centroid = None
        self.lost_frames   = 0
        self.tracker_ok    = False

    def _new_tracker(self):
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            return cv2.TrackerCSRT_create()

    def update(self, frame):
        if frame is None:
            return self.last_centroid, 'lost'

        centroid, bbox = find_object_by_color(frame, self.target_color)

        if centroid is not None:
            self.lost_frames   = 0
            self.last_centroid = centroid
            self.tracker       = self._new_tracker()
            self.tracker.init(frame, bbox)
            self.tracker_ok    = True
            return centroid, 'detected'

        if self.tracker_ok and self.tracker is not None:
            ok, bbox = self.tracker.update(frame)
            if ok:
                x, y, w, h = [int(v) for v in bbox]
                cx = x + w // 2
                cy = y + h // 2
                self.lost_frames   = 0
                self.last_centroid = (cx, cy)
                return (cx, cy), 'tracked'
            else:
                self.tracker_ok = False

        self.lost_frames += 1
        if self.lost_frames <= TRACKER_TIMEOUT and self.last_centroid is not None:
            return self.last_centroid, 'lost_recovering'

        return None, 'lost'

    def reset(self):
        self.tracker       = None
        self.last_centroid = None
        self.lost_frames   = 0
        self.tracker_ok    = False


class LaserTracker:
    def __init__(self):
        self.tracker = None
        self.last_centroid = None
        self.tracker_ok = False
        self.lost_frames = 0

    def initialize(self, frame, pixel, box_size=40):
        u, v = pixel

        x = max(0, u - box_size // 2)
        y = max(0, v - box_size // 2)

        w = min(box_size, frame.shape[1] - x)
        h = min(box_size, frame.shape[0] - y)

        try:
            self.tracker = cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            self.tracker = cv2.TrackerCSRT_create()

        self.tracker.init(frame, (x, y, w, h))

        self.last_centroid = (u, v)
        self.tracker_ok = True


def upscale(frame):
    """Scale native 424x240 frame up to display size."""
    if frame is None:
        return np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
    return cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)


def draw_tracking_status(frame, centroid, status, scale=1.0):
    """Draw centroid indicator. scale=2.0 if drawing on upscaled frame."""
    if centroid is None or frame is None:
        return
    colors = {
        'detected':        (0,   255,   0),
        'tracked':         (0,   200, 255),
        'lost_recovering': (0,   165, 255),
        'lost':            (0,     0, 255),
    }
    color = colors.get(status, (255, 255, 255))
    sc = (int(centroid[0] * scale), int(centroid[1] * scale))
    cv2.circle(frame, sc, int(12 * scale), color, 2)
    cv2.circle(frame, sc, int(3  * scale), color, -1)
    cv2.putText(frame, status.upper(), (sc[0]+14, sc[1]),
                cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)


def main():
    print("Starting sim, please wait...")
    controller.get_time()
    print("Ready. I/K=tilt  J/L=pan  B=blue box  R=red cylinder\n")

    transforms = RobotTransforms(controller)

    stow_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "wrist_yaw_counterclockwise": 0.0,
        "wrist_pitch_up": 0.0,
        "gripper_open": 0.3,
        "arm_out": 0.0,
    })

    pre_grip_pose = StateController(controller, {
        "wrist_roll_counterclockwise": 0.0,
        "gripper_open": 0.4,
    })

    Kp_pan     = 1.0
    Kp_tilt    = 1.0
    Kp_angle   = 5.0 / math.pi
    Kp_forward = 2.0
    Kp_lift    = 5.0
    Kp_yaw     = 0.5
    Kp_pitch   = 0.5
    Kp_arm     = 15.0

    phase            = "waiting"
    in_zone          = False
    target_color     = None
    head_tracker     = None
    wrist_tracker    = None
    reach_start_time = None
    recent_errors    = None

    print(f"Phase: {phase}")

    try:
        while True:
            try:
                rgb_head  = HEAD_RGB_CAMERA.get_frame()
                rgb_wrist = WRIST_RGB_CAMERA.get_frame()
            except ConnectionError:
                print("\nSimulator closed, exiting.")
                break

            velocities      = teleop.get_normalized_velocities()
            auto_velocities = {}

            # -- WAITING ---------------------------------------------------
            if phase == "waiting":
                display = upscale(rgb_head)

                if rgb_head is None:
                    cv2.putText(display, "Camera warming up...",
                                (20, DISP_H//2), cv2.FONT_HERSHEY_DUPLEX,
                                0.7, (100, 100, 100), 1, cv2.LINE_AA)

                # HUD bar
                cv2.rectangle(display, (0, 0), (DISP_W, 34), (20, 20, 20), -1)
                cv2.putText(display,
                            "I/K = tilt    J/L = pan    B = blue box    R = red cylinder    Q = quit",
                            (8, 23), cv2.FONT_HERSHEY_DUPLEX, 0.42,
                            (0, 220, 255), 1, cv2.LINE_AA)

                # Highlight detected objects (scaled to display coords)
                if rgb_head is not None:
                    sx = DISP_W / rgb_head.shape[1]
                    sy = DISP_H / rgb_head.shape[0]
                    for color, bgr in [('blue', (255, 100, 0)), ('red', (0, 0, 255))]:
                        c, _ = find_object_by_color(rgb_head, color)
                        if c is not None:
                            dc = (int(c[0]*sx), int(c[1]*sy))
                            cv2.circle(display, dc, 18, bgr, 2)
                            cv2.putText(display,
                                        "BLUE" if color == 'blue' else "RED",
                                        (dc[0]+20, dc[1]),
                                        cv2.FONT_HERSHEY_DUPLEX, 0.5,
                                        bgr, 1, cv2.LINE_AA)

                cv2.imshow("Select Target", display)
                key = cv2.waitKey(1) & 0xFF

                tilt_vel = 0.0
                pan_vel  = 0.0
                if key == ord('i'): tilt_vel =  HEAD_VEL
                if key == ord('k'): tilt_vel = -HEAD_VEL
                if key == ord('j'): pan_vel  =  HEAD_VEL
                if key == ord('l'): pan_vel  = -HEAD_VEL
                controller.set_velocities({
                    'head_tilt_up':              tilt_vel,
                    'head_pan_counterclockwise': pan_vel,
                })

                if key == ord('q'):
                    break
                elif key == ord('b'):
                    target_color  = 'blue'
                    head_tracker  = RobustTracker('blue')
                    wrist_tracker = RobustTracker('blue')
                    phase = "approach"
                    controller.set_velocities({})
                    cv2.destroyWindow("Select Target")
                    print(f"Selected: blue box ? Phase: {phase}")
                elif key == ord('r'):
                    target_color  = 'red'
                    head_tracker  = RobustTracker('red')
                    wrist_tracker = RobustTracker('red')
                    phase = "approach"
                    controller.set_velocities({})
                    cv2.destroyWindow("Select Target")
                    print(f"Selected: red cylinder ? Phase: {phase}")

                velocities = merge_proportional(velocities, stow_pose.get_command())

            # -- APPROACH --------------------------------------------------
            elif phase == "approach":
                centroid, status = head_tracker.update(rgb_head)

                if centroid is not None and rgb_head is not None:
                    cx, cy = centroid
                    frame_cx = rgb_head.shape[1] / 2
                    frame_cy = rgb_head.shape[0] / 2

                    if status in ('detected', 'tracked'):
                        error_x = (cx - frame_cx) / rgb_head.shape[1]
                        error_y = (cy - frame_cy) / rgb_head.shape[0]
                        auto_velocities["head_pan_counterclockwise"] = -Kp_pan * error_x
                        auto_velocities["head_tilt_up"] = -Kp_tilt * error_y

                    _, obj2base_T = locate_object((cx, cy), HEAD_CAMERA, transforms)
                    if obj2base_T is not None:
                        x, y, z = obj2base_T[0:3, 3]
                        angle_z = math.atan2(y, x)
                        horizontal_distance = math.sqrt(x**2 + y**2)

                        cam_z = transforms.get_wrist_cam_T()[2, 3]
                        auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01) + 0.10)

                        # Wider zone entry/exit for more consistent triggering
                        if not in_zone:
                            if 0.40 <= horizontal_distance <= 0.60:
                                in_zone = True
                        else:
                            if horizontal_distance < 0.35 or horizontal_distance > 0.65:
                                in_zone = False

                        if in_zone:
                            angle_error = -math.pi / 2 - angle_z
                            auto_velocities["base_counterclockwise"] = Kp_angle * angle_error
                            auto_velocities["base_forward"] = 0.0
                            print(f"\rDist: {horizontal_distance:.2f}m  "
                                  f"Angle: {math.degrees(angle_error):+.1f}°  [{status}] [FLANK]   ",
                                  end="", flush=True)
                            if abs(angle_error) < math.radians(5):
                                cv2.destroyWindow("Head RGB")
                                phase = "align"
                                wrist_tracker.reset()
                                print(f"\nPhase: {phase}")
                        else:
                            auto_velocities["base_counterclockwise"] = -Kp_angle * angle_z
                            alignment = 1.0 - (abs(angle_z) / math.pi)

                            # Lower threshold so robot moves forward sooner (was 0.9)
                            travel_auth = max(0.0, min(1.0, (alignment - 0.7) / 0.3))

                            # Cap forward speed to avoid overshooting
                            auto_velocities["base_forward"] = (
                                Kp_forward * min(horizontal_distance, 0.5) * travel_auth)

                            print(f"\rDist: {horizontal_distance:.2f}m  "
                                  f"Align: {alignment:.3f}  [{status}] [moving]   ",
                                  end="", flush=True)

                # Draw on upscaled frame
                if rgb_head is not None:
                    disp = upscale(rgb_head)
                    sx = DISP_W / rgb_head.shape[1]
                    sy = DISP_H / rgb_head.shape[0]
                    if centroid:
                        draw_tracking_status(disp, centroid, status, scale=min(sx, sy))
                    fcx = int(rgb_head.shape[1]/2 * sx)
                    fcy = int(rgb_head.shape[0]/2 * sy)
                    cv2.circle(disp, (fcx, fcy), 6, (0, 0, 255), 2)
                    cv2.imshow("Head RGB", disp)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

            # -- ALIGN -----------------------------------------------------
            elif phase == "align":
                centroid, status = wrist_tracker.update(rgb_wrist)

                if centroid is not None:
                    cx, cy = centroid
                    _, obj2base_T = locate_object((cx, cy), WRIST_CAMERA, transforms)
                    if obj2base_T is not None:
                        x, y, z = obj2base_T[0:3, 3]
                        angle_z = math.atan2(y, x)
                        cam_z = transforms.get_wrist_cam_T()[2, 3]

                        auto_velocities["base_counterclockwise"] = Kp_angle * (
                            -math.pi / 2 - angle_z + math.radians(3))
                        auto_velocities["lift_up"] = Kp_lift * (z - (cam_z + 0.01))

                        angle_err = abs(-math.pi / 2 - angle_z)
                        lift_err  = abs(z - (cam_z + 0.01))
                        if stow_pose.is_at_goal() and angle_err < math.radians(5) and lift_err < 0.03:
                            phase = "reach"
                            print(f"\nPhase: {phase}")

                if rgb_wrist is not None:
                    disp = upscale(rgb_wrist)
                    if centroid:
                        sx = DISP_W / rgb_wrist.shape[1]
                        sy = DISP_H / rgb_wrist.shape[0]
                        draw_tracking_status(disp, centroid, status, scale=min(sx, sy))
                    cv2.imshow("Wrist RGB", disp)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, stow_pose.get_command())

            # -- REACH -----------------------------------------------------
            elif phase == "reach":
                # Initialize reach tracking variables on phase entry
                if reach_start_time is None:
                    reach_start_time = time.time()
                if recent_errors is None:
                    recent_errors = []

                centroid, status = wrist_tracker.update(rgb_wrist)

                if centroid is not None:
                    cx, cy = centroid
                    if rgb_wrist is not None:
                        frame_cx = rgb_wrist.shape[1] / 2
                        frame_cy = rgb_wrist.shape[0] / 2
                        if status in ('detected', 'tracked'):
                            error_x = (cx - frame_cx) / rgb_wrist.shape[1]
                            error_y = (cy - frame_cy) / rgb_wrist.shape[0]
                            auto_velocities["wrist_yaw_counterclockwise"] = Kp_yaw * error_x
                            auto_velocities["wrist_pitch_up"] = -Kp_pitch * error_y

                    distance = WRIST_CAMERA.get_depth((cx, cy))
                    if distance is not None:
                        distance_error = distance - 0.12

                        # Rolling average over last 10 frames
                        recent_errors.append(abs(distance_error))
                        if len(recent_errors) > 10:
                            recent_errors.pop(0)
                        avg_error = sum(recent_errors) / len(recent_errors)

                        timed_out = (time.time() - reach_start_time) > 10.0

                        auto_velocities["arm_out"] = Kp_arm * distance_error
                        print(f"\rDist: {distance:.3f}m  Err: {distance_error:+.3f}m  "
                              f"Avg: {avg_error:.3f}m  [{status}]{'  [TIMEOUT]' if timed_out else ''}   ",
                              end="", flush=True)

                        # Relaxed to 0.025 (2.5cm) — tight enough for precision,
                        # loose enough to not rely on timeout every time
                        if (avg_error < 0.025 or timed_out) and pre_grip_pose.is_at_goal():
                            cv2.destroyAllWindows()
                            phase = "grab"
                            reach_start_time = None  # reset for next run
                            recent_errors = None
                            print(f"\nPhase: {phase}")

                if rgb_wrist is not None:
                    disp = upscale(rgb_wrist)
                    if centroid:
                        sx = DISP_W / rgb_wrist.shape[1]
                        sy = DISP_H / rgb_wrist.shape[0]
                        draw_tracking_status(disp, centroid, status, scale=min(sx, sy))
                    cv2.imshow("Wrist RGB", disp)

                velocities = merge_proportional(velocities, auto_velocities)
                velocities = merge_proportional(velocities, pre_grip_pose.get_command())

            # -- GRAB ------------------------------------------------------
            elif phase == "grab":
                auto_velocities["lift_up"] = 0.2
                auto_velocities["gripper_open"] = -1.0
                velocities = merge_proportional(velocities, auto_velocities)
                print("\rGrabbing...   ", end="", flush=True)

            controller.set_velocities(velocities)
            cv2.waitKey(1)
            time.sleep(1 / 30)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        try:
            controller.set_velocities({})
            controller.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()
        print("Done.")


if __name__ == "__main__":
    main()
