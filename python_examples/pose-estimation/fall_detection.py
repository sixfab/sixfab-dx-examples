"""App 2/3: Fall Detection - multi-person.

Every detected person is tracked independently (nearest-centroid
tracking), each with their own torso-angle/fall-speed history and alert
state - so it can watch a room with several people, not just one.

Two signals combined per person to reduce false positives (sitting/
crouching down on purpose should NOT trigger this):
  1. Torso angle from vertical (shoulder-midpoint -> hip-midpoint line).
     Upright ~0 deg, lying down ~90 deg.
  2. Vertical fall speed: how fast that person's hip midpoint dropped over
     the last ~0.6s. A real fall is fast; sitting down slowly is not.

A fall is flagged when the torso goes mostly horizontal (> FALL_ANGLE_DEG)
AND a fast drop (> FALL_SPEED_FRAC_PER_SEC of the frame height per second)
happened within the last FALL_DROP_MEMORY_SEC seconds. Once
flagged for a person, their alert stays up until a cooldown expires (or
'r' clears all alerts), so it doesn't spam while they're still down.

Keys: 'q' quit, 'r' clear all alerts.
"""
import sys
import time
from collections import deque

import cv2
import numpy as np

from pose_engine import (
    FpsCounter, PoseEngine, draw_text, open_source, read_looping, update_pose_tracks,
    L_SHOULDER, R_SHOULDER, L_HIP, R_HIP,
)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"

# Torso angle from vertical above which the body counts as "mostly horizontal".
# 55 was too low: a person kneeling and bending over someone reaches ~60 deg
# and, with the fast hip drop of kneeling, was flagged as a fall.
FALL_ANGLE_DEG = 70.0
# A person must be upright again (below this angle) before they can trigger
# another alert, so someone lying on the floor raises one alert, not many.
REARM_ANGLE_DEG = 40.0
# A person often drops fast while the torso is still upright, and only then
# turns horizontal. So a fast hip drop is remembered for this many seconds:
# if the torso is mostly horizontal within that window, it counts as a fall.
FALL_DROP_MEMORY_SEC = 2.0
# Tracking match distance as a fraction of the frame width. A falling body's
# box centre jumps far between frames, and the default 150 px broke the track.
TRACK_MAX_DIST_FRAC = 0.3
# Fall speed as a fraction of the frame height per second, so the threshold
# does not depend on resolution. 0.55 equals the original 400 px/s at the
# 720p frame height the apps request from a camera (400 / 720).
FALL_SPEED_FRAC_PER_SEC = 0.55
HISTORY_SECONDS = 0.6
ALERT_COOLDOWN_SEC = 5.0

engine = PoseEngine()
cap, is_camera = open_source(SOURCE)
fps_counter = FpsCounter()

WINDOW = "Fall Detection (q=quit r=clear alerts)"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)

tracks = []

# Time base: a live camera uses the wall clock. A video file uses the video's
# own clock (frame number / fps), so the fall speed does not change when the
# Pi processes frames slower or faster than the file's frame rate.
src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frames_read = 0

print("Fall Detection ready. Works for multiple people at once.", flush=True)


def new_state():
    return {"hip_y_history": deque(), "alert_until": 0.0, "armed": True, "last_fast_drop": -1e9}


while True:
    ok, frame = read_looping(cap, is_camera)
    if not ok:
        break
    if is_camera:
        frame = cv2.flip(frame, 1)
    frames_read += 1
    now = time.time() if is_camera else frames_read / src_fps

    poses = engine.infer(frame)
    tracks = update_pose_tracks(tracks, poses, max_dist=TRACK_MAX_DIST_FRAC * frame.shape[1])
    any_alert = False

    for t in tracks:
        if t.age > 0:
            continue
        pose = t.pose
        engine.draw_skeleton(frame, pose)
        st = t.data.setdefault("fall", new_state())

        ls, rs = pose.kp(L_SHOULDER), pose.kp(R_SHOULDER)
        lh, rh = pose.kp(L_HIP), pose.kp(R_HIP)
        shoulders = [p for p in (ls, rs) if p]
        hips = [p for p in (lh, rh) if p]

        torso_angle = None
        fall_speed = 0.0
        if shoulders and hips:
            sx = sum(p[0] for p in shoulders) / len(shoulders)
            sy = sum(p[1] for p in shoulders) / len(shoulders)
            hx = sum(p[0] for p in hips) / len(hips)
            hy = sum(p[1] for p in hips) / len(hips)

            dx, dy = hx - sx, hy - sy
            torso_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

            hist = st["hip_y_history"]
            hist.append((now, hy))
            while hist and now - hist[0][0] > HISTORY_SECONDS:
                hist.popleft()
            if len(hist) >= 2:
                t0, y0 = hist[0]
                dt = now - t0
                if dt > 0.05:
                    fall_speed = (hy - y0) / dt / frame.shape[0]  # frame heights per second

        if fall_speed > FALL_SPEED_FRAC_PER_SEC:
            st["last_fast_drop"] = now
        if torso_angle is not None and torso_angle < REARM_ANGLE_DEG:
            st["armed"] = True
        if (st["armed"] and torso_angle is not None and torso_angle > FALL_ANGLE_DEG
                and now - st["last_fast_drop"] <= FALL_DROP_MEMORY_SEC):
            st["armed"] = False
            st["alert_until"] = now + ALERT_COOLDOWN_SEC
            print(f"Person #{t.id}: FALL DETECTED (angle={torso_angle:.0f} deg, speed={fall_speed:.2f} frame heights/s)", flush=True)

        x1, y1, x2, y2 = [int(v) for v in pose.bbox]
        is_alert = now < st["alert_until"]
        any_alert = any_alert or is_alert
        box_color = (0, 0, 255) if is_alert else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3 if is_alert else 1)
        label = f"#{t.id}"
        if torso_angle is not None:
            label += f"  angle={torso_angle:.0f}"
        draw_text(frame, label, (x1, max(20, y1 - 10)), scale=0.6, color=box_color)
        if is_alert:
            draw_text(frame, "FALL!", (x1, y2 + 24), scale=0.9, color=(0, 0, 255), thickness=3)

    if any_alert:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.15, frame, 0.85, 0)
        draw_text(frame, "FALL DETECTED", (frame.shape[1] // 2 - 180, 50),
                  scale=1.3, color=(0, 0, 255), thickness=3)

    fps = fps_counter.tick()
    draw_text(frame, f"People tracked: {sum(1 for t in tracks if t.age == 0)}", (12, frame.shape[0] - 20), scale=0.7)
    draw_text(frame, f"{fps:.1f} FPS", (frame.shape[1] - 140, 30), scale=0.6)
    if not poses:
        draw_text(frame, "No pose detected - step into frame", (12, 40), color=(0, 165, 255))

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        for t in tracks:
            t.data["fall"] = new_state()

cap.release()
cv2.destroyAllWindows()
print("Closed.")
