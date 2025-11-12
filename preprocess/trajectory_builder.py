import os
import numpy as np
from .detector import DetectorStub
from .tracker import TrackerStub

def crop_region(img, bbox, pad=0):
    x1,y1,x2,y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, x1-pad); y1 = max(0, y1-pad)
    x2 = min(w, x2+pad); y2 = min(h, y2+pad)
    if x2<=x1 or y2<=y1:
        return np.zeros((1,1,3), dtype=img.dtype)
    return img[y1:y2, x1:x2, :]

def extract_appearance_feat(patch):
    # very simple: global avg pooling of RGB (normalize to float)
    p = patch.astype(np.float32) / 255.0
    return p.mean(axis=(0,1))  # returns 3-dim feature

def build_and_cache_trajectories(frames, ops=None, out_npz=None):
    """
    frames: list of images (H,W,3) as numpy arrays
    ops: optional list of op events (frame_idx, x,y, type)
    out_npz: optional path to save cached .npz
    returns: trajectories list (dict serializable)
    """
    detector = DetectorStub()
    tracker = TrackerStub()
    per_frame_dets = []
    tracker.reset()
    for i, img in enumerate(frames):
        dets = detector.detect(img)
        per_frame_dets.append(dets)
        tracker.update(dets, i)
    tracks = tracker.export_tracks()
    # build structured trajectories
    trajectories = []
    for t in tracks:
        traj = {"id": t["id"], "frames": []}
        for obs in t["observations"]:
            fidx = obs["frame_idx"]
            bbox = obs["bbox"]
            patch = crop_region(frames[fidx], bbox, pad=4)
            feat = extract_appearance_feat(patch)
            traj["frames"].append({"frame_idx": fidx, "bbox": bbox, "feat": feat.tolist()})
        # compute deltas
        for k in range(1, len(traj["frames"])):
            a = np.array(traj["frames"][k]["feat"])
            b = np.array(traj["frames"][k-1]["feat"])
            traj["frames"][k]["delta"] = (a - b).tolist()
        trajectories.append(traj)
    if out_npz:
        np.savez_compressed(out_npz, trajectories=trajectories)
    return trajectories

if __name__ == "__main__":
    # tiny smoke test: create 3 synthetic frames and save cache
    import cv2
    h,w = 240,320
    frames = []
    for i in range(3):
        img = np.ones((h,w,3), dtype=np.uint8) * (50 + i*10)
        frames.append(img)
    out = build_and_cache_trajectories(frames, out_npz="sample_traj.npz")
    print("Saved sample_traj.npz, num trajectories:", len(out))
