import cv2
import numpy as np

def draw_tracks_on_frame(frame, tracks):
    """
    frame: H,W,3
    tracks: list of dict {id, frames: list of {frame_idx, bbox}}
    draws last bbox of each track
    """
    out = frame.copy()
    for t in tracks:
        if len(t.get("frames", []))==0: continue
        last = t["frames"][-1]
        x1,y1,x2,y2 = last["bbox"]
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(out, f"id:{t['id']}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0),1)
    return out
