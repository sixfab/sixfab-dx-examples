"""App 1/3: Pilates Coach - arm-press rep counter with form feedback.

Multi-person: every detected person gets their own tracked rep counter
(nearest-centroid tracking keeps each person's count stable across
frames), drawn above their own bounding box - useful for a class with
several people in frame, not just a single user.

Tracks the elbow angle (shoulder-elbow-wrist), averaging both arms when
visible. Per person, state machine: EXTENDED -> BENT -> EXTENDED counts
one rep; the minimum angle reached during the BENT phase drives the form
feedback.

Keys: 'q' quit, 'r' reset all rep counters.
"""
import sys
import time

import cv2

from pose_engine import (
    FpsCounter, PoseEngine, angle, draw_text, open_source, read_looping,
    update_pose_tracks, L_SHOULDER, L_ELBOW, L_WRIST, R_SHOULDER, R_ELBOW, R_WRIST,
)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"

EXTENDED_ANGLE = 150.0   # elbow angle above this = arms extended (press out)
BENT_ANGLE = 110.0       # elbow angle below this = arms bent (pulled in)
GOOD_BEND_ANGLE = 100.0  # min angle must reach this or lower for a "good" rep

engine = PoseEngine()
cap, is_camera = open_source(SOURCE)
fps_counter = FpsCounter()

WINDOW = "Pilates Coach - Arm Press Counter (q=quit r=reset)"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)

tracks = []

print("Pilates Coach ready. Make sure shoulders, elbows and wrists are visible - works for multiple people at once.", flush=True)


def new_state():
    return {"state": "extended", "reps": 0, "min_angle": 180.0, "feedback": "", "feedback_until": 0.0}


while True:
    ok, frame = read_looping(cap, is_camera)
    if not ok:
        break
    if is_camera:
        frame = cv2.flip(frame, 1)

    poses = engine.infer(frame)
    tracks = update_pose_tracks(tracks, poses)

    for t in tracks:
        if t.age > 0:
            continue  # not matched this frame, don't draw/update stale data
        pose = t.pose
        engine.draw_skeleton(frame, pose)
        st = t.data.setdefault("fitness", new_state())

        left = angle(pose.kp(L_SHOULDER), pose.kp(L_ELBOW), pose.kp(L_WRIST))
        right = angle(pose.kp(R_SHOULDER), pose.kp(R_ELBOW), pose.kp(R_WRIST))
        sides = [a for a in (left, right) if a is not None]
        elbow_angle = sum(sides) / len(sides) if sides else None

        if elbow_angle is not None:
            if st["state"] == "extended" and elbow_angle < BENT_ANGLE:
                st["state"] = "bent"
                st["min_angle"] = elbow_angle
            elif st["state"] == "bent":
                st["min_angle"] = min(st["min_angle"], elbow_angle)
                if elbow_angle > EXTENDED_ANGLE:
                    st["state"] = "extended"
                    st["reps"] += 1
                    if st["min_angle"] <= GOOD_BEND_ANGLE:
                        st["feedback"] = f"Nice control! ({st['min_angle']:.0f} deg)"
                    else:
                        st["feedback"] = f"Pull in further ({st['min_angle']:.0f} deg)"
                    st["feedback_until"] = time.time() + 2.5
                    print(f"Person #{t.id} - REP {st['reps']}: {st['feedback']}", flush=True)

        x1, y1, x2, y2 = [int(v) for v in pose.bbox]
        label = f"#{t.id}  Reps: {st['reps']}"
        draw_text(frame, label, (x1, max(20, y1 - 12)), scale=0.7, color=(0, 255, 0))
        if time.time() < st["feedback_until"]:
            draw_text(frame, st["feedback"], (x1, y2 + 22), scale=0.6, color=(0, 255, 255))

    fps = fps_counter.tick()
    draw_text(frame, f"People tracked: {sum(1 for t in tracks if t.age == 0)}", (12, 40), scale=0.9)
    draw_text(frame, f"{fps:.1f} FPS", (frame.shape[1] - 140, 30), scale=0.6)
    if not poses:
        draw_text(frame, "No pose detected - step into frame", (12, 75), color=(0, 165, 255))

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        for t in tracks:
            t.data["fitness"] = new_state()

cap.release()
cv2.destroyAllWindows()
total = sum(t.data.get("fitness", {}).get("reps", 0) for t in tracks)
print(f"Closed. Tracked {len(tracks)} people, total reps across all: {total}")
