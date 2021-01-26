import json
import time

import numpy as np

FAKE_FACE_FILE = ''
scale = 1.0
frame_count = 0

class FakeFaceDet:
    frame_count = 0
    def __init__(self):
        with open(FAKE_FACE_FILE, 'r') as f:
            bbox_dict = json.load(f)
        self.bboxes = bbox_dict['bboxes']
        self.previous_call_time = 0

    def detect(self, img):
        global frame_count
        t = int(time.time() * 1000)
        diff = int(time.time() * 1000) - self.previous_call_time
        diff = max(30 - diff, 0)
        self.previous_call_time = t
        time.sleep(diff/1000.)
        if len(self.bboxes) <= frame_count:
            return []
        box = self.bboxes[frame_count]
        box = np.array(box).astype(np.float32) * scale
        frame_count += 1
        return box.astype(np.int32).tolist()
