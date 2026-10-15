"""Retail analytics - live preview window: yolox-s_640x640_ppu.dxnn (DeepX
Model Zoo, pretrained COCO-80) on the local DX-M1 NPU. Custom pipeline (no
Frigate) - person detection + centroid tracking + entry/exit line-crossing
counter + foot-traffic heatmap, in two OpenCV windows:
  1) live feed with a decaying "recent activity" heatmap overlay
  2) a non-decaying "total" heatmap, redrawn once every HEATMAP_TOTAL_PERIOD_SEC

Keys: 'q' quit, 'h' toggle live heatmap overlay, 'r' reset counters.
On quit, the total heatmap is saved to disk.
"""
import os
import sys
import time

import cv2
import numpy as np
from dx_engine import InferenceEngine, InferenceOption

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DXNN_PATH = os.path.join(OUT_DIR, "model", "yolox-s_640x640_ppu.dxnn")
VIDEO_IN = sys.argv[1] if len(sys.argv) > 1 else "0"

PERSON_CLASS = 0  # COCO-80

PPU_BOX_BYTES = (0, 16)
PPU_GRID_BYTES = (16, 20)
PPU_SCORE_BYTES = (20, 24)
PPU_LABEL_BYTES = (24, 28)
YOLOX_STRIDES = np.array([8, 16, 32], dtype=np.float32)  # multi-scale anchor-free head, grid-relative
THRESHOLD = 0.45
NMS_THRESHOLD = 0.45

WORK_WIDTH = 1280  # downscale the (4K) source to this before any drawing/heatmap work - big speedup
LINE_Y_FRAC = 0.62
TRACK_MAX_DIST = 90
TRACK_MAX_AGE = 15
HEATMAP_DECAY = 0.98
HEATMAP_TOTAL_PERIOD_SEC = 60

opts = InferenceOption()
opts.devices = [0]
engine = InferenceEngine(DXNN_PATH, opts)
side = engine.get_input_tensors_info()[0]["shape"][1]
print(f"Model loaded (input {side}x{side}). Keys: q=quit h=heatmap r=reset")


def reinterpret(tensor, byte_range, dtype):
    lo, hi = byte_range
    return np.ascontiguousarray(tensor[:, lo:hi]).view(dtype)


def letterbox(img, size, pad_value=114):
    ih, iw = img.shape[:2]
    scale = min(size / iw, size / ih)
    nw, nh = int(round(iw * scale)), int(round(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), pad_value, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top : top + nh, left : left + nw] = resized
    return canvas, scale, left, top


def decode_persons(records, pad_left, pad_top, scale, w, h):
    """Vectorized YOLOX PPU decode, person class only. Returns boxes_xywh (for
    NMS), scores, and (x1,y1,x2,y2) boxes in the working-frame coordinates."""
    if len(records) == 0:
        return [], [], []

    scores = reinterpret(records, PPU_SCORE_BYTES, np.float32).flatten()
    keep = scores > THRESHOLD
    if not keep.any():
        return [], [], []
    sel, scores = records[keep], scores[keep]

    labels = reinterpret(sel, PPU_LABEL_BYTES, np.uint32).flatten()
    keep = labels == PERSON_CLASS
    if not keep.any():
        return [], [], []
    sel, scores = sel[keep], scores[keep]

    grid = reinterpret(sel, PPU_GRID_BYTES, np.uint8)
    grid_y, grid_x = grid[:, 0].astype(np.float32), grid[:, 1].astype(np.float32)
    layer_idx = np.clip(grid[:, 3].astype(np.int32), 0, len(YOLOX_STRIDES) - 1)
    stride = YOLOX_STRIDES[layer_idx]

    boxes = reinterpret(sel, PPU_BOX_BYTES, np.float32).reshape(-1, 4)
    cx = (boxes[:, 0] + grid_x) * stride
    cy = (boxes[:, 1] + grid_y) * stride
    bw = np.exp(boxes[:, 2]) * stride
    bh = np.exp(boxes[:, 3]) * stride
    mx, my = cx - bw / 2, cy - bh / 2

    x1 = np.clip((mx - pad_left) / scale, 0, w)
    y1 = np.clip((my - pad_top) / scale, 0, h)
    x2 = np.clip((mx + bw - pad_left) / scale, 0, w)
    y2 = np.clip((my + bh - pad_top) / scale, 0, h)

    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    boxes_orig = np.stack([x1, y1, x2, y2], axis=1).astype(int).tolist()
    return boxes_xywh, scores.tolist(), boxes_orig


class Track:
    _next_id = 1

    def __init__(self, centroid):
        self.id = Track._next_id
        Track._next_id += 1
        self.centroid = centroid
        self.prev_centroid = centroid
        self.age = 0
        self.counted = False


def update_tracks(tracks, detections):
    unmatched_dets = list(range(len(detections)))
    for t in tracks:
        if not unmatched_dets:
            break
        dists = [np.hypot(*(detections[i] - t.centroid)) for i in unmatched_dets]
        j = int(np.argmin(dists))
        if dists[j] < TRACK_MAX_DIST:
            di = unmatched_dets.pop(j)
            t.prev_centroid = t.centroid
            t.centroid = detections[di]
            t.age = 0
        else:
            t.age += 1
    for i in unmatched_dets:
        tracks.append(Track(detections[i]))
    return [t for t in tracks if t.age <= TRACK_MAX_AGE]


def colorize_heatmap(hmap):
    if hmap.max() <= 0:
        return np.zeros((*hmap.shape, 3), np.uint8)
    norm = (hmap / hmap.max() * 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_JET)


WINDOW = "Retail Analytics - live (q=quit h=heatmap r=reset)"
WINDOW_TOTAL = f"Retail Analytics - TOTAL heatmap (updates every {HEATMAP_TOTAL_PERIOD_SEC}s)"
cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
cv2.namedWindow(WINDOW_TOTAL, cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(int(VIDEO_IN) if VIDEO_IN.isdigit() else VIDEO_IN)
src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
work_h = int(src_h * (WORK_WIDTH / src_w))
w, h = WORK_WIDTH, work_h
line_y = int(h * LINE_Y_FRAC)
cv2.resizeWindow(WINDOW, w, h)
cv2.resizeWindow(WINDOW_TOTAL, w, h)

heatmap_live = np.zeros((h, w), dtype=np.float32)
heatmap_total = np.zeros((h, w), dtype=np.float32)
total_display = colorize_heatmap(heatmap_total)
last_total_update = time.time()

tracks: list[Track] = []
entries = exits = 0
show_heatmap = False  # starts off - the heatmap tint darkens the plain view, press 'h' to enable
fps_t0, fps_n = time.time(), 0
fps = 0.0

while True:
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    padded, scale, pad_left, pad_top = letterbox(rgb, side)
    x = np.expand_dims(padded, axis=0)

    out = engine.run([x])[0]
    arr = np.asarray(out)
    records = arr[0] if arr.ndim == 3 and arr.shape[0] > 0 else np.zeros((0, 32), np.uint8)

    boxes_xywh, scores, boxes_orig = decode_persons(records, pad_left, pad_top, scale, w, h)

    keep_idx = []
    if boxes_xywh:
        keep = cv2.dnn.NMSBoxes(boxes_xywh, scores, THRESHOLD, NMS_THRESHOLD)
        keep_idx = np.array(keep).reshape(-1).tolist()

    detections = []
    for i in keep_idx:
        x1, y1, x2, y2 = boxes_orig[i]
        detections.append(np.array([(x1 + x2) / 2, (y1 + y2) / 2]))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    tracks = update_tracks(tracks, detections)

    for t in tracks:
        if t.age > 0:
            continue
        cx, cy = t.centroid
        ix, iy = int(cx), int(cy)
        heatmap_live[max(0, iy - 15) : iy + 15, max(0, ix - 15) : ix + 15] += 1.0
        heatmap_total[max(0, iy - 15) : iy + 15, max(0, ix - 15) : ix + 15] += 1.0
        cv2.circle(frame, (ix, iy), 4, (0, 0, 255), -1)
        cv2.putText(frame, f"#{t.id}", (ix + 6, iy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        if not t.counted:
            py, ny = t.prev_centroid[1], cy
            if py < line_y <= ny:
                entries += 1
                t.counted = True
            elif py > line_y >= ny:
                exits += 1
                t.counted = True

    heatmap_live *= HEATMAP_DECAY

    if show_heatmap:
        frame = cv2.addWeighted(frame, 0.75, colorize_heatmap(heatmap_live), 0.35, 0)
    cv2.line(frame, (0, line_y), (w, line_y), (255, 0, 255), 2)

    fps_n += 1
    if time.time() - fps_t0 >= 1.0:
        fps = fps_n / (time.time() - fps_t0)
        fps_t0, fps_n = time.time(), 0

    stats = [
        f"In frame: {len(detections)}",
        f"Entries: {entries}   Exits: {exits}   Occupancy: {entries - exits}",
        f"{fps:.1f} FPS",
    ]
    for i, line in enumerate(stats):
        y = 30 + i * 28
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.imshow(WINDOW, frame)

    now = time.time()
    if now - last_total_update >= HEATMAP_TOTAL_PERIOD_SEC:
        total_display = colorize_heatmap(heatmap_total)
        last_total_update = now
    label = f"TOTAL since start - next refresh in {HEATMAP_TOTAL_PERIOD_SEC - int(now - last_total_update)}s"
    total_frame = total_display.copy()
    cv2.putText(total_frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(total_frame, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    cv2.imshow(WINDOW_TOTAL, total_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("h"):
        show_heatmap = not show_heatmap
    elif key == ord("r"):
        entries = exits = 0
        heatmap_live[:] = 0

cap.release()
cv2.destroyAllWindows()

final_total = colorize_heatmap(heatmap_total)
out_path = os.path.join(OUT_DIR, "total_heatmap_final.png")
cv2.imwrite(out_path, final_total)
print(f"Closed. Final: entries={entries} exits={exits} occupancy={entries - exits}")
print(f"Total heatmap saved: {out_path}")
