"""
laser_dot_test.py — Laser pointer simulator for the Stretch robot.

Controls  (click the Head Camera window first for keyboard focus)
─────────────────────────────────────────────────────────────────
  I / K       Tilt camera up / down       (only when not locked)
  J / L       Pan camera left / right     (only when not locked)
  SPACE       Lock laser dot onto crosshair (manual selection)
  A           Toggle AUTO mode — detects physical red laser pointer
  R           Release lock — re-enables camera movement
  C           Clear laser dot entirely
  P           Print current pan / tilt angles
  Q           Quit

AUTO MODE (press A)
───────────────────
  When enabled, the robot continuously scans the camera frame for a
  physical red laser pointer dot and locks onto it automatically.
  The dot must be:
    • Very bright red (Value ≥ 200 in HSV)
    • Highly saturated (Saturation ≥ 150 in HSV)
    • Very small (blob area < 300 px²)
  Tune these thresholds inside laser_dot.py → update_from_frame().
"""

import cv2
import threading
import time
from stretch_toolkit import HEAD_CAMERA, HEAD_RGB_CAMERA, HEAD_DEPTH_CAMERA, controller
from stretch_mujoco.enums.actuators import Actuators
from laser_dot import LaserDotSimulator

# ── constants ────────────────────────────────────────────────────────────────
WINDOW             = "Stretch  |  Head Camera"
NATIVE_W, NATIVE_H = 424, 240
DISP_W,   DISP_H   = 848, 480
CX, CY             = NATIVE_W // 2, NATIVE_H // 2
HEAD_VEL           = 2.0 #er = snappier response

# ── colours (BGR) ────────────────────────────────────────────────────────────
C_CROSSHAIR  = (0,   220, 255)
C_LOCKED_RG  = (0,   200,  80)
C_AUTO_COLOR = (0,   180, 255)   # cyan label for AUTO mode
C_HUD_VAL    = (255, 255, 255)
C_HUD_LBL    = (140, 140, 140)
C_KEY_HI     = (0,   220, 255)
C_BAR        = (20,   20,  20)
C_LOCK_BAR   = (0,    60,  20)
C_AUTO_BAR   = (0,    40,  60)   # dark cyan bar when auto is on


# ── key state (shared between threads) ───────────────────────────────────────
_keys_held = set()
_keys_lock = threading.Lock()


def draw_ui(frame, pan, tilt, dot_pixel, dot_world, locked, auto_mode):
    h, w = frame.shape[:2]

    sx, sy = CX * 2, CY * 2
    if not locked:
        gap, arm = 14, 24
        cv2.line(frame, (sx-arm, sy), (sx-gap, sy), C_CROSSHAIR, 1)
        cv2.line(frame, (sx+gap, sy), (sx+arm, sy), C_CROSSHAIR, 1)
        cv2.line(frame, (sx, sy-arm), (sx, sy-gap), C_CROSSHAIR, 1)
        cv2.line(frame, (sx, sy+gap), (sx, sy+arm), C_CROSSHAIR, 1)
        cv2.circle(frame, (sx, sy), gap-2, C_CROSSHAIR, 1)

    if dot_pixel is not None:
        du, dv = dot_pixel[0]*2, dot_pixel[1]*2
        cv2.circle(frame, (du, dv), 13, (255,255,255), 2)
        cv2.circle(frame, (du, dv),  8, (0, 0, 220),  -1)
        cv2.circle(frame, (du-2, dv-2), 2, (255,255,255), -1)
        lx = min(du+16, w-90)
        ly = max(dv-10, 40)
        label = "AUTO" if auto_mode else "LOCKED"
        color = C_AUTO_COLOR if auto_mode else C_LOCKED_RG
        cv2.putText(frame, label, (lx, ly),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)

    # Top bar colour depends on mode
    if auto_mode:
        bar_color = C_AUTO_BAR
    elif locked:
        bar_color = C_LOCK_BAR
    else:
        bar_color = C_BAR

    cv2.rectangle(frame, (0, 0), (w, 30), bar_color, -1)
    cv2.putText(frame, "PAN",         (10,  20), cv2.FONT_HERSHEY_DUPLEX, 0.48, C_HUD_LBL, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{pan:+.2f}", (46,  20), cv2.FONT_HERSHEY_DUPLEX, 0.48, C_HUD_VAL, 1, cv2.LINE_AA)
    cv2.putText(frame, "TILT",        (118, 20), cv2.FONT_HERSHEY_DUPLEX, 0.48, C_HUD_LBL, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{tilt:+.2f}",(158, 20), cv2.FONT_HERSHEY_DUPLEX, 0.48, C_HUD_VAL, 1, cv2.LINE_AA)

    if auto_mode:
        cv2.putText(frame, "AUTO LASER ON  |  A=off",
                    (255, 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, C_AUTO_COLOR, 1, cv2.LINE_AA)
    elif dot_world:
        x, y, z = dot_world
        cv2.putText(frame, f"TARGET  x={x:+.2f}  y={y:+.2f}  z={z:.2f} m",
                    (255, 20), cv2.FONT_HERSHEY_DUPLEX, 0.45, C_LOCKED_RG, 1, cv2.LINE_AA)

    if locked and not auto_mode:
        cv2.putText(frame, "LOCKED  |  R=release  C=clear",
                    (w-290, 20), cv2.FONT_HERSHEY_DUPLEX, 0.42, C_LOCKED_RG, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, h-24), (w, h), C_BAR, -1)
    hints = [
        ("I/K","tilt"), ("J/L","pan"), ("SPACE","select"),
        ("A","auto"),   ("R","release"), ("C","clear"), ("Q","quit"),
    ]
    xc = 10
    for key, lbl in hints:
        key_color = C_AUTO_COLOR if (key == "A" and auto_mode) else C_KEY_HI
        cv2.putText(frame, key, (xc, h-7), cv2.FONT_HERSHEY_DUPLEX, 0.40, key_color, 1, cv2.LINE_AA)
        kw = cv2.getTextSize(key, cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)[0][0]
        cv2.putText(frame, f" {lbl}", (xc+kw, h-7), cv2.FONT_HERSHEY_DUPLEX, 0.40, C_HUD_LBL, 1, cv2.LINE_AA)
        lw = cv2.getTextSize(f" {lbl}", cv2.FONT_HERSHEY_DUPLEX, 0.40, 1)[0][0]
        xc += kw + lw + 16


def velocity_thread(stop_event, locked_ref):
    """
    Runs at 50Hz independently of the camera frame rate.
    Sends head velocity commands as long as keys are held.
    This decouples key responsiveness from render lag.
    """
    while not stop_event.is_set():
        with _keys_lock:
            keys = set(_keys_held)

        if not locked_ref[0]:  # only move when not locked
            tilt_vel = 0.0
            pan_vel  = 0.0

            if ord('i') in keys: tilt_vel =  HEAD_VEL
            if ord('k') in keys: tilt_vel = -HEAD_VEL
            if ord('j') in keys: pan_vel  =  HEAD_VEL
            if ord('l') in keys: pan_vel  = -HEAD_VEL

            controller.set_velocities({
                'head_tilt_up':              tilt_vel,
                'head_pan_counterclockwise': pan_vel,
            })
        else:
            # Locked — ensure head is stopped
            controller.set_velocities({
                'head_tilt_up': 0.0,
                'head_pan_counterclockwise': 0.0,
            })

        time.sleep(1 / 50)


def main():
    print("Starting sim, please wait...")
    controller.get_time()
    print("Sim connected.")
    print("  SPACE  → manual lock on crosshair")
    print("  A      → toggle AUTO laser pointer detection")
    print("  I/K/J/L → move camera")
    print("  R/C/Q  → release / clear / quit\n")

    laser      = LaserDotSimulator(HEAD_CAMERA, window_name=WINDOW)
    locked_ref = [False]   # mutable so velocity thread can read it
    auto_mode  = [False]   # auto laser detection toggle
    stop_event = threading.Event()
    space_held = False

    # Start velocity thread — handles key→movement at 50Hz
    vthread = threading.Thread(target=velocity_thread,
                               args=(stop_event, locked_ref), daemon=True)
    vthread.start()

    try:
        while True:
            rgb   = HEAD_RGB_CAMERA.get_frame()
            depth = HEAD_DEPTH_CAMERA.get_frame()   # watchdog keep-alive

            if rgb is None:
                continue

            # ── AUTO laser detection ──────────────────────────────────────
            if auto_mode[0] and depth is not None:
                laser.update_from_frame(rgb, depth)

            dot_px    = laser.get_dot_pixel()
            dot_world = laser.get_dot_world()
            locked_ref[0] = dot_px is not None

            # Read live pan/tilt for HUD display
            status = controller.sim.pull_status()
            pan    = status.head_pan.pos
            tilt   = status.head_tilt.pos

            display = cv2.resize(rgb, (DISP_W, DISP_H), interpolation=cv2.INTER_LINEAR)
            draw_ui(display, pan, tilt, dot_px, dot_world, locked_ref[0], auto_mode[0])
            cv2.imshow(WINDOW, display)

            if laser.dot_updated():
                w = laser.get_dot_world()
                src = "AUTO" if auto_mode[0] else "CLICK"
                print(f"  [{src}] TARGET → ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:.3f}) m")

            key = cv2.waitKey(1) & 0xFF

            # Track held keys for velocity thread
            with _keys_lock:
                if key != 0xFF:
                    _keys_held.add(key)
                if key == 0xFF:
                    _keys_held.clear()

            if key == ord('q'):
                break
            elif key == ord('p'):
                print(f"  pan={pan:.3f}  tilt={tilt:.3f}")
            elif key == ord('a'):
                auto_mode[0] = not auto_mode[0]
                state = "ON" if auto_mode[0] else "OFF"
                print(f"  Auto laser detection: {state}")
                if not auto_mode[0]:
                    laser.clear()
                    locked_ref[0] = False
            elif key == ord('c'):
                laser.clear()
                locked_ref[0] = False
                print("  Dot cleared.")
            elif key == ord('r') and locked_ref[0]:
                laser.clear()
                locked_ref[0] = False
                print("  Released.")
            elif key == ord(' ') and not space_held and not locked_ref[0]:
                space_held = True
                if depth is None:
                    print("  [!] Depth not ready, try again.")
                else:
                    laser.select_pixel(CX, CY, depth)

            if key != ord(' '):
                space_held = False

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        controller.set_velocities({'head_tilt_up': 0.0, 'head_pan_counterclockwise': 0.0})
        cv2.destroyAllWindows()
        controller.stop()
        print("Done.")


if __name__ == "__main__":
    main()
