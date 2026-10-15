"""Headless benchmark for the pose engine on a video file.

Usage: python benchmark.py <video-file> [<video-file> ...]

For each file it reads 30 warm-up frames, then 300 timed frames (looping the
file if it is shorter), and prints per-frame timings for:
  - PoseEngine.infer(): preprocess + NPU run + decode + NMS
  - one full loop: read + infer + tracking (no window, no drawing)
Run it with nothing else using the NPU, and note the board and DX-RT version.
"""
import os, sys, time, statistics as st
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_engine import PoseEngine, update_pose_tracks

def run(path, frames=300, warmup=30):
    eng = PoseEngine()
    cap = cv2.VideoCapture(path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    infer_ms, loop_ms, npose = [], [], []
    tracks = []
    n = 0
    while n < frames + warmup:
        t0 = time.perf_counter()
        ok, f = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, f = cap.read()
        t1 = time.perf_counter()
        poses = eng.infer(f)
        t2 = time.perf_counter()
        tracks = update_pose_tracks(tracks, poses, max_dist=0.3 * f.shape[1])
        t3 = time.perf_counter()
        n += 1
        if n > warmup:
            infer_ms.append((t2 - t1) * 1000)
            loop_ms.append((t3 - t0) * 1000)
            npose.append(len(poses))
    q = lambda a, p: sorted(a)[int(len(a) * p) - 1]
    tot = sum(loop_ms) / 1000
    print(f"{path.split('/')[-1]}: {W}x{H} @ {src_fps:.0f} fps source, {frames} frames after {warmup} warm-up")
    print(f"  infer() per frame (preprocess + NPU + decode + NMS): mean {st.mean(infer_ms):.1f} ms, median {st.median(infer_ms):.1f} ms, p95 {q(infer_ms,0.95):.1f} ms")
    print(f"  read + infer + tracking, headless:                    mean {st.mean(loop_ms):.1f} ms  ->  {frames/tot:.1f} FPS")
    print(f"  poses per frame: mean {st.mean(npose):.2f}, max {max(npose)}")

for p in sys.argv[1:]:
    run(p)
