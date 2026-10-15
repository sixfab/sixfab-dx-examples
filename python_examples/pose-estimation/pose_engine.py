"""Shared pose-estimation engine: yolov8n-pose.dxnn on the local DX-M1 NPU.

Reads directly from a USB camera via cv2.VideoCapture (no RTSP - latency
matters for the interactive apps built on top of this). Raw model output
(no PPU): (1, 56, 8400) = 4 box + 1 conf + 17*3 keypoints per anchor,
decoded and NMS'd in Python (same approach validated in
an earlier validation run: 82.5 FPS, 94% hit rate).
"""
import os
import time

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

DXNN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "yolov8n-pose.dxnn")

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45

# Duplicate suppression after NMS. A person lying down has a wide, flat box,
# and the model sometimes returns a second, weaker copy of the same person
# whose box mostly sits inside the stronger one. NMS keeps it because the
# IoU is below NMS_THRESHOLD. A pose is dropped when its box lies at least
# DUP_CONTAINMENT inside a stronger pose's box AND its own confidence is
# below DUP_MAX_CONF. A confident person standing in front of or behind
# another person is kept.
DUP_CONTAINMENT = 0.7
DUP_MAX_CONF = 0.6

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW = 5, 6, 7, 8
L_WRIST, R_WRIST, L_HIP, R_HIP = 9, 10, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
]


def letterbox(img, size, pad_value=114):
    ih, iw = img.shape[:2]
    scale = min(size / iw, size / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


class Pose:
    __slots__ = ("bbox", "conf", "keypoints")

    def __init__(self, bbox, conf, keypoints):
        self.bbox = bbox  # (x1, y1, x2, y2) in original frame coords
        self.conf = conf
        self.keypoints = keypoints  # (17, 3) array: x, y, visibility, in original frame coords

    def kp(self, idx, min_vis=0.5):
        """Returns (x, y) for a keypoint index, or None if not confidently visible."""
        x, y, v = self.keypoints[idx]
        return (float(x), float(y)) if v > min_vis else None


class PoseEngine:
    def __init__(self, model_path=DXNN_PATH, conf_threshold=CONF_THRESHOLD, nms_threshold=NMS_THRESHOLD):
        opts = InferenceOption()
        opts.devices = [0]
        self.engine = InferenceEngine(model_path, opts)
        self.side = self.engine.get_input_tensors_info()[0]["shape"][1]
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def infer(self, frame_bgr) -> list[Pose]:
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        padded, scale, pad_left, pad_top = letterbox(rgb, self.side)
        x = np.expand_dims(padded, axis=0)

        out = self.engine.run([x])[0]
        pred = np.array(out).reshape(56, 8400).T
        boxes_cxcywh = pred[:, :4]
        conf = pred[:, 4]
        kpts = pred[:, 5:].reshape(-1, 17, 3)

        mask = conf > self.conf_threshold
        if not mask.any():
            return []

        boxes_f, conf_f, kpts_f = boxes_cxcywh[mask], conf[mask], kpts[mask]
        x1 = boxes_f[:, 0] - boxes_f[:, 2] / 2
        y1 = boxes_f[:, 1] - boxes_f[:, 3] / 2
        boxes_xywh = np.column_stack([x1, y1, boxes_f[:, 2], boxes_f[:, 3]])

        keep = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), conf_f.tolist(), self.conf_threshold, self.nms_threshold)
        poses = []
        for i in np.array(keep).reshape(-1):
            bx, by, bw, bh = boxes_xywh[i]
            bx1 = max(0.0, (bx - pad_left) / scale)
            by1 = max(0.0, (by - pad_top) / scale)
            bx2 = min(float(w), (bx + bw - pad_left) / scale)
            by2 = min(float(h), (by + bh - pad_top) / scale)

            kp_orig = kpts_f[i].copy()
            kp_orig[:, 0] = (kp_orig[:, 0] - pad_left) / scale
            kp_orig[:, 1] = (kp_orig[:, 1] - pad_top) / scale

            poses.append(Pose((bx1, by1, bx2, by2), float(conf_f[i]), kp_orig))
        return self._drop_duplicates(poses)

    @staticmethod
    def _drop_duplicates(poses):
        """Removes weak poses whose box is mostly inside a stronger pose's box."""
        poses = sorted(poses, key=lambda p: p.conf, reverse=True)
        kept = []
        for p in poses:
            ax1, ay1, ax2, ay2 = p.bbox
            area = max(1e-6, (ax2 - ax1) * (ay2 - ay1))
            duplicate = False
            if p.conf < DUP_MAX_CONF:
                for q in kept:
                    bx1, by1, bx2, by2 = q.bbox
                    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
                    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
                    if iw * ih / area >= DUP_CONTAINMENT:
                        duplicate = True
                        break
            if not duplicate:
                kept.append(p)
        return kept

    def draw_skeleton(self, frame, pose: Pose, color=(255, 128, 0), point_color=(0, 0, 255)):
        pts = [pose.kp(i) for i in range(17)]
        for a, b in SKELETON:
            if pts[a] and pts[b]:
                cv2.line(frame, (int(pts[a][0]), int(pts[a][1])), (int(pts[b][0]), int(pts[b][1])), color, 3)
        for p in pts:
            if p:
                cv2.circle(frame, (int(p[0]), int(p[1])), 5, point_color, -1)


def angle(a, b, c):
    """Angle at vertex b, formed by points a-b-c, in degrees. Any of a/b/c may be None."""
    if a is None or b is None or c is None:
        return None
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return None
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def open_usb_camera(index=0, width=640, height=480):
    """Opens a USB camera directly (DSHOW backend on Windows avoids the
    slow default MSMF init), requesting a higher resolution than the
    device default if supported.

    Forces manual exposure mode: on this camera (and many cheap UVC
    webcams), DSHOW's "auto exposure" mode silently caps capture at
    ~15fps regardless of requested resolution/fps - switching to manual
    mode at the SAME exposure value doubles it to ~30fps. Confirmed via
    diagnose_capture_backends.py: this is a driver quirk, not a real
    lighting/exposure-time limitation."""
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency - don't queue stale frames
    if os.name == "nt":
        current_exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DSHOW manual-mode flag
        cap.set(cv2.CAP_PROP_EXPOSURE, current_exposure)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open USB camera at index {index}")
    return cap


def open_source(spec, width=1280, height=720):
    """Opens either a USB camera (spec is an int-like camera index, e.g.
    "0") or a video file (spec is a path). Returns (cap, is_camera)."""
    is_camera = str(spec).isdigit()
    if is_camera:
        return open_usb_camera(int(spec), width, height), True
    cap = cv2.VideoCapture(str(spec))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {spec}")
    return cap, False


def read_looping(cap, is_camera):
    """Reads one frame; if this is a file (not a live camera) and it hit
    the end, loops back to the start instead of returning False."""
    ok, frame = cap.read()
    if not ok and not is_camera:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    return ok, frame


class FpsCounter:
    def __init__(self):
        self.t0 = time.time()
        self.n = 0
        self.fps = 0.0

    def tick(self):
        self.n += 1
        dt = time.time() - self.t0
        if dt >= 1.0:
            self.fps = self.n / dt
            self.t0, self.n = time.time(), 0
        return self.fps


class PoseTrack:
    """A tracked person across frames, with a stable id and a free-form
    `data` dict apps can use for their own per-person state (rep counter,
    fall-alert state, etc)."""
    _next_id = 1

    def __init__(self, pose: Pose):
        self.id = PoseTrack._next_id
        PoseTrack._next_id += 1
        self.pose = pose
        self.age = 0
        self.data = {}

    def centroid(self):
        x1, y1, x2, y2 = self.pose.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def update_pose_tracks(tracks: list[PoseTrack], poses: list[Pose], max_dist=150, max_age=40) -> list[PoseTrack]:
    """Nearest-centroid multi-person tracking, same approach as the
    centroid tracker in the retail-analytics project. Keeps each person's
    `.data` state stable across frames as long as they don't move too far
    or disappear for too many frames in a row.

    Matching is global, not per track: every (track, pose) pair closer than
    max_dist is sorted by distance and assigned from the closest pair up, so
    the result does not depend on track order. Every track that gets no pose
    this frame ages by one, and is dropped after max_age frames. A long max_age
    matters when two people overlap: the model can return one merged skeleton
    for a few frames, and each person's track must survive that gap to get its
    own detection back (with a short max_age the ids swapped). (The earlier
    per-track loop stopped early and left unmatched tracks at age 0, so their
    old skeletons stayed on screen.)"""

    def centroid_of(pose: Pose):
        x1, y1, x2, y2 = pose.bbox
        return np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    pose_centroids = [centroid_of(p) for p in poses]
    pairs = []
    for ti, t in enumerate(tracks):
        tc = np.array(t.centroid())
        for pi, pc in enumerate(pose_centroids):
            d = float(np.hypot(*(pc - tc)))
            if d < max_dist:
                pairs.append((d, ti, pi))
    pairs.sort()

    used_tracks, used_poses = set(), set()
    for d, ti, pi in pairs:
        if ti in used_tracks or pi in used_poses:
            continue
        used_tracks.add(ti)
        used_poses.add(pi)
        tracks[ti].pose = poses[pi]
        tracks[ti].age = 0

    for ti, t in enumerate(tracks):
        if ti not in used_tracks:
            t.age += 1

    for pi, pose in enumerate(poses):
        if pi not in used_poses:
            tracks.append(PoseTrack(pose))

    return [t for t in tracks if t.age <= max_age]


def draw_text(frame, text, pos, scale=0.8, color=(255, 255, 255), thickness=2):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
