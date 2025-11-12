import numpy as np

class DetectorStub:
    """
    Very simple detector stub for unit tests / prototyping.
    It returns one or two dummy bbox proposals covering the image.
    Replace with Detectron2 / YOLOv8 in production.
    """
    def __init__(self):
        pass

    def detect(self, img):
        # img: numpy H,W,3
        h, w = img.shape[:2]
        # return list of dicts: bbox [x1,y1,x2,y2], score, cls
        return [
            {"bbox": [int(w*0.1), int(h*0.1), int(w*0.4), int(h*0.4)], "score": 0.9, "cls": "widget"},
            {"bbox": [int(w*0.5), int(h*0.2), int(w*0.9), int(h*0.6)], "score": 0.8, "cls": "button"}
        ]
