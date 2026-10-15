"""App 3/3: Fruit-Ninja-style body-controlled game.

Both wrists act as "blades" - their recent trail is drawn as a glowing
slash, and any fruit the trail passes through gets sliced (removed,
scored, small burst effect). Fruits launch from the bottom with gravity,
arcing up and falling back down, same as real Fruit Ninja. Missing a
fruit (it falls past the bottom uncut) costs a life; 3 misses ends the
round.

Keys: 'q' quit, 'r' restart after game over.
"""
import random
import sys
import time
from collections import deque

import cv2
import numpy as np

from pose_engine import (
    FpsCounter, PoseEngine, draw_text, open_source, read_looping, update_pose_tracks,
    L_WRIST, R_WRIST,
)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"
BLADE_COLORS = [(255, 220, 0), (0, 220, 255), (255, 0, 200), (0, 255, 120)]

GRAVITY = 1400.0          # px/s^2
SPAWN_INTERVAL = (0.5, 1.2)  # seconds, randomized between spawns
FRUIT_RADIUS = 38
TRAIL_SECONDS = 0.18
TRAIL_MAX_POINTS = 10
MAX_LIVES = 3

FRUIT_COLORS = [
    (60, 60, 220),   # apple - red (BGR)
    (60, 165, 255),  # orange
    (60, 220, 220),  # lemon
    (90, 200, 60),   # lime
]


class Fruit:
    def __init__(self, w, h):
        self.x = random.uniform(w * 0.15, w * 0.85)
        self.y = float(h + FRUIT_RADIUS)
        self.vx = random.uniform(-120, 120)
        self.vy = -random.uniform(1.15, 1.5) * h * 1.15  # launched higher
        self.color = random.choice(FRUIT_COLORS)
        self.radius = FRUIT_RADIUS
        self.sliced = False

    def update(self, dt):
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt

    def draw(self, frame):
        c = (int(self.x), int(self.y))
        cv2.circle(frame, c, self.radius, self.color, -1)
        cv2.circle(frame, c, self.radius, (255, 255, 255), 2)
        cv2.circle(frame, (c[0] - self.radius // 3, c[1] - self.radius // 3), self.radius // 4, (255, 255, 255), -1)


class SliceEffect:
    def __init__(self, x, y, color):
        self.x, self.y, self.color = x, y, color
        self.t0 = time.time()

    def alive(self):
        return time.time() - self.t0 < 0.35

    def draw(self, frame):
        age = time.time() - self.t0
        r = int(20 + age * 160)
        alpha_radius = max(0, FRUIT_RADIUS + 25 - r)
        if alpha_radius <= 0:
            return
        cv2.circle(frame, (int(self.x), int(self.y)), r, self.color, 3)


def point_segment_distance(p, a, b):
    p, a, b = np.array(p, float), np.array(a, float), np.array(b, float)
    ab = b - a
    denom = np.dot(ab, ab)
    if denom < 1e-6:
        return np.linalg.norm(p - a)
    t = np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


class Blade:
    def __init__(self, color):
        self.points = deque()  # (x, y, t)
        self.color = color
        self.last_pos = None  # always the latest known hand position, for a persistent marker
        self.last_seen = 0.0

    def update(self, pos):
        now = time.time()
        if pos is not None:
            self.points.append((pos[0], pos[1], now))
            self.last_pos = pos
            self.last_seen = now
        while self.points and now - self.points[0][2] > TRAIL_SECONDS:
            self.points.popleft()
        while len(self.points) > TRAIL_MAX_POINTS:
            self.points.popleft()

    def draw(self, frame):
        pts = list(self.points)
        for i in range(1, len(pts)):
            x1, y1, _ = pts[i - 1]
            x2, y2, _ = pts[i]
            thickness = max(2, int(2 + 6 * i / max(1, len(pts))))
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.color, thickness, cv2.LINE_AA)
        if self.last_pos is not None and time.time() - self.last_seen < 1.0:
            p = (int(self.last_pos[0]), int(self.last_pos[1]))
            cv2.circle(frame, p, 14, self.color, -1)
            cv2.circle(frame, p, 14, (255, 255, 255), 2)

    def hits(self, fruit):
        pts = list(self.points)
        for i in range(1, len(pts)):
            a, b = pts[i - 1][:2], pts[i][:2]
            if point_segment_distance((fruit.x, fruit.y), a, b) < fruit.radius:
                return True
        if self.last_pos is not None and time.time() - self.last_seen < 1.0:
            dx, dy = fruit.x - self.last_pos[0], fruit.y - self.last_pos[1]
            if (dx * dx + dy * dy) ** 0.5 < fruit.radius:
                return True
        return False


def new_game():
    return {
        "fruits": [], "effects": [], "score": 0, "lives": MAX_LIVES,
        "next_spawn": time.time() + 0.5, "game_over": False,
    }


engine = PoseEngine()
cap, is_camera = open_source(SOURCE)
fps_counter = FpsCounter()

WINDOW = "Fruit Ninja - Pose Edition (q=quit r=restart)"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW, 1280, 720)

tracks = []
game = new_game()
last_t = time.time()

print("Fruit Ninja (pose edition) ready. Everyone in frame can slice - multiplayer!", flush=True)

while True:
    ok, frame = read_looping(cap, is_camera)
    if not ok:
        break
    if is_camera:
        frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    now = time.time()
    dt = min(0.05, now - last_t)
    last_t = now

    poses = engine.infer(frame)
    tracks = update_pose_tracks(tracks, poses)

    all_blades = []
    for t in tracks:
        if t.age > 0:
            continue
        color = BLADE_COLORS[t.id % len(BLADE_COLORS)]
        blades = t.data.setdefault("blades", {"L": Blade(color), "R": Blade(color)})
        blades["L"].update(t.pose.kp(L_WRIST))
        blades["R"].update(t.pose.kp(R_WRIST))
        all_blades.append(blades["L"])
        all_blades.append(blades["R"])

    if not game["game_over"]:
        if now >= game["next_spawn"]:
            game["fruits"].append(Fruit(w, h))
            game["next_spawn"] = now + random.uniform(*SPAWN_INTERVAL)

        for fruit in game["fruits"]:
            fruit.update(dt)
            if not fruit.sliced and any(blade.hits(fruit) for blade in all_blades):
                fruit.sliced = True
                game["score"] += 1
                game["effects"].append(SliceEffect(fruit.x, fruit.y, fruit.color))

        survivors = []
        for fruit in game["fruits"]:
            if fruit.sliced:
                continue
            if fruit.y - fruit.radius > h:
                game["lives"] -= 1
                if game["lives"] <= 0:
                    game["game_over"] = True
                continue
            survivors.append(fruit)
        game["fruits"] = survivors

    game["effects"] = [e for e in game["effects"] if e.alive()]

    for fruit in game["fruits"]:
        fruit.draw(frame)
    for effect in game["effects"]:
        effect.draw(frame)
    for blade in all_blades:
        blade.draw(frame)

    fps = fps_counter.tick()
    draw_text(frame, f"Score: {game['score']}", (12, 40), scale=1.0, color=(0, 255, 0))
    draw_text(frame, f"Lives: {'X ' * game['lives']}".strip(), (12, 75), color=(0, 0, 255))
    draw_text(frame, f"Players: {sum(1 for t in tracks if t.age == 0)}", (12, 105), scale=0.7)
    draw_text(frame, f"{fps:.1f} FPS", (w - 140, 30), scale=0.6)
    if not poses:
        draw_text(frame, "No pose detected - step into frame", (12, h - 20), color=(0, 165, 255))

    if game["game_over"]:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        draw_text(frame, "GAME OVER", (w // 2 - 150, h // 2 - 20), scale=1.6, color=(0, 0, 255), thickness=3)
        draw_text(frame, f"Final score: {game['score']}", (w // 2 - 130, h // 2 + 30), scale=1.0)
        draw_text(frame, "Press 'r' to play again", (w // 2 - 140, h // 2 + 65), scale=0.8)

    cv2.imshow(WINDOW, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r") and game["game_over"]:
        game = new_game()

cap.release()
cv2.destroyAllWindows()
print(f"Closed. Final score: {game['score']}")
