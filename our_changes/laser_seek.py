

"""
laser_seek.py — Autonomous laser pointer seeker for the Stretch robot.

BEHAVIOUR
─────────
  SCAN   → Head sweeps a pan/tilt grid looking for the red laser dot.
            If nothing found after a full sweep, restarts from the beginning.
  TRACK  → Dot found — head servo-centres on it at ~30 Hz.
            Once centred (error < CENTRE_TOL px), transitions to LOCKED.
  LOCKED → Dot position is fixed. Head holds still.
            If the dot disappears for LOST_FRAMES consecutive frames,
            returns to SCAN automatically.

INTEGRATION (drop-in alongside laser_dot.py)
────────────────────────────────────────────
    from laser_dot  import LaserDotSimulator
    from laser_seek import LaserSeeker

    laser  = LaserDotSimulator(HEAD_CAMERA, window_name="Head Camera")
    seeker = LaserSeeker(laser, controller)

    # In your main loop (call every frame):
    rgb   = HEAD_RGB_CAMERA.get_frame()
    depth = HEAD_DEPTH_CAMERA.get_frame()
    seeker.update(rgb, depth)

    # Read state:
    state = seeker.state          # "scan" | "track" | "locked"
    world = laser.get_dot_world() # 3-D position when locked

TUNING
──────
  SCAN_PAN_RANGE   pan limits for sweep  [rad]
  SCAN_TILT_RANGE  tilt limits for sweep [rad]
  SCAN_PAN_STEP    angular step per column
  SCAN_TILT_STEP   angular step per row
  SCAN_DWELL       seconds to wait at each grid position before moving on
  TRACK_Kp         proportional gain for centring servo (rad/px)
  CENTRE_TOL       pixel radius considered "centred"
  LOST_FRAMES      consecutive missed frames before giving up lock
"""

import time
import math
import threading
import numpy as np
from stretch_mujoco.enums.actuators import Actuators

# ── Sweep grid ────────────────────────────────────────────────────────────────
SCAN_PAN_RANGE  = (-1.4,  1.4)    # rad  left … right
SCAN_TILT_RANGE = (-0.8,  0.1)    # rad  down  … level
SCAN_PAN_STEP   = 0.30            # rad  ~17°
SCAN_TILT_STEP  = 0.25            # rad  ~14°
SCAN_DWELL      = 0.30            # s    settle + grab frame at each grid pos

# ── Tracking servo ────────────────────────────────────────────────────────────
TRACK_Kp    = 0.003               # rad per pixel
CENTRE_TOL  = 18                  # px   — considered centred when error < this
TRACK_MAX_V = 0.6                 # rad/s cap on head velocity during tracking

# ── Lock / re-acquire ─────────────────────────────────────────────────────────
LOST_FRAMES = 8                   # missed detection frames before returning to SCAN


# ─────────────────────────────────────────────────────────────────────────────

class LaserSeeker:
    """
    Autonomous pan/tilt search + servo-centre behaviour for a red laser dot.

    Parameters
    ----------
    laser : LaserDotSimulator
        Instance that handles detection and 3-D unprojection.
    controller :
        stretch_toolkit controller (has .sim.move_to / .set_velocities /
        .sim.pull_status).
    native_w, native_h : int
        Native camera frame dimensions (default 424 × 240).
    """

    def __init__(self, laser, controller,
                 native_w: int = 424, native_h: int = 240):
        self._laser      = laser
        self._ctrl       = controller
        self._native_w   = native_w
        self._native_h   = native_h

        # Build sweep grid: list of (pan, tilt) waypoints in a boustrophedon
        # (snake) pattern so the head moves continuously without large jumps.
        self._grid   = self._build_grid()
        self._grid_i = 0          # current waypoint index

        self._state       = "scan"
        self._lost_count  = 0
        self._dwell_until = 0.0   # timestamp: stay at grid pos until this

        # Cached joint positions (read from sim status)
        self._pan  = 0.0
        self._tilt = 0.0

        print("[LaserSeeker] Initialised — starting in SCAN mode.")
        print(f"  Grid: {len(self._grid)} waypoints  "
              f"pan {SCAN_PAN_RANGE}  tilt {SCAN_TILT_RANGE}")

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def state(self) -> str:
        """Current seeker state: 'scan', 'track', or 'locked'."""
        return self._state

    def reset(self):
        """Force back to SCAN (e.g. after user clears the dot)."""
        self._laser.clear()
        self._state      = "scan"
        self._grid_i     = 0
        self._lost_count = 0
        print("[LaserSeeker] Reset → SCAN")

    def update(self, rgb_frame, depth_frame) -> str:
        """
        Call every frame from your main loop.

        Runs one tick of the state machine, issues head velocity commands,
        and calls laser.update_from_frame() for detection.

        Returns
        -------
        str
            Current state after the tick: 'scan' | 'track' | 'locked'.
        """
        # Refresh cached joint angles
        try:
            status     = self._ctrl.sim.pull_status()
            self._pan  = status.head_pan.pos
            self._tilt = status.head_tilt.pos
        except Exception:
            pass

        if self._state == "scan":
            self._tick_scan(rgb_frame, depth_frame)
        elif self._state == "track":
            self._tick_track(rgb_frame, depth_frame)
        elif self._state == "locked":
            self._tick_locked(rgb_frame, depth_frame)

        return self._state

    # ── State ticks ───────────────────────────────────────────────────────────

    def _tick_scan(self, rgb, depth):
        """Move head through the sweep grid; run detection at every position."""

        # Try to detect even while moving — opportunistic
        detected = self._laser.update_from_frame(rgb, depth)
        if detected:
            self._stop_head()
            self._state      = "track"
            self._lost_count = 0
            print(f"[LaserSeeker] SCAN → TRACK  (grid pos {self._grid_i})")
            return

        now = time.time()

        # Still dwelling at current waypoint?
        if now < self._dwell_until:
            self._stop_head()
            return

        # Move to next waypoint
        if self._grid_i >= len(self._grid):
            self._grid_i = 0   # restart sweep

        target_pan, target_tilt = self._grid[self._grid_i]
        self._ctrl.sim.move_to(Actuators.head_pan,  target_pan)
        self._ctrl.sim.move_to(Actuators.head_tilt, target_tilt)
        self._dwell_until = now + SCAN_DWELL
        self._grid_i += 1

    def _tick_track(self, rgb, depth):
        """
        Servo the head so the laser dot sits at the frame centre.
        Transition to LOCKED when centred; back to SCAN if lost.
        """
        detected = self._laser.update_from_frame(rgb, depth)

        if not detected:
            self._lost_count += 1
            if self._lost_count >= LOST_FRAMES:
                self._stop_head()
                self._state  = "scan"
                self._grid_i = 0
                print("[LaserSeeker] TRACK → SCAN  (dot lost)")
            return

        self._lost_count = 0
        pixel = self._laser.get_dot_pixel()
        if pixel is None:
            return

        u, v = pixel
        cx   = self._native_w / 2
        cy   = self._native_h / 2
        eu   = u - cx   # positive → dot is to the right
        ev   = v - cy   # positive → dot is below centre

        # Proportional velocity commands to centre the head on the dot
        # Pan:  dot right → pan right (negative vel in "counterclockwise" axis)
        # Tilt: dot below → tilt down (negative vel in "up" axis)
        pan_vel  = float(np.clip(-TRACK_Kp * eu * 60,  # ×60 → rad/s approx
                                  -TRACK_MAX_V, TRACK_MAX_V))
        tilt_vel = float(np.clip(-TRACK_Kp * ev * 60,
                                  -TRACK_MAX_V, TRACK_MAX_V))

        self._ctrl.set_velocities({
            "head_pan_counterclockwise": pan_vel,
            "head_tilt_up":              tilt_vel,
        })

        err_px = math.hypot(eu, ev)
        if err_px < CENTRE_TOL:
            self._stop_head()
            self._state = "locked"
            w = self._laser.get_dot_world()
            print(f"[LaserSeeker] TRACK → LOCKED  "
                  f"pixel=({u},{v})  world={_fmt(w)}")

    def _tick_locked(self, rgb, depth):
        """
        Hold the lock. Re-detect every frame to keep the world coord fresh.
        If dot disappears, return to SCAN.
        """
        self._stop_head()
        detected = self._laser.update_from_frame(rgb, depth)

        if not detected:
            self._lost_count += 1
            if self._lost_count >= LOST_FRAMES:
                self._laser.clear()
                self._state  = "scan"
                self._grid_i = 0
                print("[LaserSeeker] LOCKED → SCAN  (dot lost)")
        else:
            self._lost_count = 0

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _stop_head(self):
        self._ctrl.set_velocities({
            "head_pan_counterclockwise": 0.0,
            "head_tilt_up":              0.0,
        })

    @staticmethod
    def _build_grid():
        """
        Build a boustrophedon (snake) grid of (pan, tilt) waypoints.
        Columns alternate direction to minimise travel distance.
        """
        tilts = _frange(SCAN_TILT_RANGE[1], SCAN_TILT_RANGE[0], -abs(SCAN_TILT_STEP))
        pans  = _frange(SCAN_PAN_RANGE[0],  SCAN_PAN_RANGE[1],   abs(SCAN_PAN_STEP))

        grid = []
        for row_i, tilt in enumerate(tilts):
            col = pans if row_i % 2 == 0 else list(reversed(pans))
            for pan in col:
                grid.append((pan, tilt))
        return grid


# ── Utilities ─────────────────────────────────────────────────────────────────

def _frange(start, stop, step):
    """Float range (inclusive of start; stops before or at stop)."""
    vals, v = [], start
    while (step > 0 and v <= stop + 1e-9) or (step < 0 and v >= stop - 1e-9):
        vals.append(round(v, 6))
        v += step
    return vals


def _fmt(world):
    if world is None:
        return "None"
    return f"({world[0]:+.3f}, {world[1]:+.3f}, {world[2]:.3f}) m"

