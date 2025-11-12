import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    boxBArea = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea/denom if denom>0 else 0.0

class TrackerStub:
    """
    Simple per-frame tracker: greedily matches detections to previous tracks by IoU.
    tracks: list of dict {id, observations: list of {frame_idx, bbox}}
    """
    def __init__(self, iou_thresh=0.3):
        self.next_id = 1
        self.tracks = []  # active tracks

    def reset(self):
        self.next_id = 1
        self.tracks = []

    def update(self, dets, frame_idx):
        # dets: list of dict with 'bbox'
        assigned = {}
        if len(self.tracks) == 0:
            # initialize tracks
            for d in dets:
                t = {"id": self.next_id, "observations":[{"frame_idx":frame_idx, "bbox": d["bbox"]}]}
                self.tracks.append(t)
                assigned[id(d)] = t["id"]
                self.next_id += 1
            return

        # build cost matrix
        used = set()
        for d in dets:
            best_iou = 0.0
            best_track = None
            for t in self.tracks:
                last_bbox = t["observations"][-1]["bbox"]
                cur_iou = iou(last_bbox, d["bbox"])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_track = t
            if best_iou >= 0.3 and best_track is not None:
                best_track["observations"].append({"frame_idx":frame_idx, "bbox": d["bbox"]})
                used.add(best_track["id"])
            else:
                # new track
                t = {"id": self.next_id, "observations":[{"frame_idx":frame_idx, "bbox": d["bbox"]}]}
                self.tracks.append(t)
                self.next_id += 1

    def export_tracks(self):
        return self.tracks
