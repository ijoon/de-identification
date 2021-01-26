import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from collections import deque


def calc_iou(bb1, bb2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    boxBArea = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class KalmanFilter:

    def __init__(self, num_mean):
        self.n = num_mean
        self.x = np.zeros(self.n)
        self.p = np.eye(self.n) * 10
        self.f = np.eye(self.n)
        self.q = np.eye(self.n) * 0.01
        self.h = np.zeros((self.n, self.n))
        np.fill_diagonal(self.h, 1)
        self.r = np.eye(self.n)
        self.r[0,0] = 0.007
        self.r[1,1] = 0.007
        self.r[2,2] = 0.1
        self.r[3,3] = 0.1

        self.x_ = np.zeros(self.n)
        self.p_ = np.zeros((self.n, self.n))

    def update(self, z):
        self.y = z - self.h.dot(self.x_).reshape(1, self.n)

        self.s = np.dot(self.h, self.p_).dot(self.h.T) + self.r
        self.k = np.dot(self.p_, self.h.T).dot(np.linalg.inv(self.s))

        self.x = self.x_ + self.k.dot(self.y.reshape(self.n, 1))
        self.p = (np.eye(self.n) - self.k.dot(self.h)).dot(self.p_)

    def predict(self):
        self.x_ = np.dot(self.f, self.x.reshape((self.n, 1)))
        self.p_ = np.dot(self.f, self.p).dot(self.f.T) + self.q
        return self.x_.reshape(-1).astype('int')

class Tracker:

    def __init__(self, id):
        self.kf = KalmanFilter(4)
        self.id = id
        self.num_hits = 0
        self.num_looses = 0

    def update(self, z):
        _ = self.kf.predict()
        self.kf.update(z)

    def predict(self):
        return self.kf.predict()

    def hit(self):
        self.num_hits += 1

    def loose(self):
        self.num_looses += 1

    def reset_hits(self):
        self.num_hits = 0

    def reset_looses(self):
        self.num_looses = 0

class TrackerPool:

    def __init__(self, num_tracker, iou_thr, patience=5):
        self.num = num_tracker
        self.iou_thr = iou_thr
        self.ids = deque([n for n in range(self.num)])
        self.trackers = []
        self.iou_mat= np.zeros((self.num, self.num), dtype=np.float32)
        self.pred_bbox = [None for n in range(self.num)]
        self.looses_patience = patience

    def _add_tracker(self, tracker):
        self.trackers.append(tracker)

    def _get_new_id(self):
        return self.ids.popleft()

    def _discard_failed_tracker(self):
        winner = []
        for t in self.trackers:
            if t.num_looses > self.looses_patience:
                self.ids.append(t.id)
            else:
                winner.append(t)
        self.trackers = winner

    def assign(self, dets):

        for r in range(len(self.trackers)):
            self.pred_bbox[r] = self.trackers[r].predict()

        iou_mat= np.zeros((len(self.trackers), len(dets)), dtype=np.float32)
        for t_idx, t in enumerate(self.trackers):
            for d_idx, d in enumerate(dets):
                iou = calc_iou(t.predict(), d)
                iou_mat[t_idx, d_idx] = iou

        matched_idx = linear_assignment(-iou_mat)

        unmatched_trackers = []
        unmatched_detections = []
        for idx in range(len(self.trackers)):
            if(idx not in matched_idx[:,0]):
                unmatched_trackers.append(idx)

        for idx in range(len(dets)):
            if(idx not in matched_idx[:,1]):
                unmatched_detections.append(idx)

        matches = []

        for m in matched_idx:
            if(iou_mat[m[0],m[1]] < self.iou_thr):
                unmatched_trackers.append(m[0])
                unmatched_detections.append(m[1])
            else:
                matches.append(m.reshape(1,2))

        if(len(matches) > 0):
            matches = np.concatenate(matches,axis=0)

        for t_idx, d_idx in matches:
            self.trackers[t_idx].update(dets[d_idx])
            self.trackers[t_idx].hit()
            self.trackers[t_idx].reset_looses()

        for idx in unmatched_detections:
            new_tracker = Tracker(self._get_new_id())
            new_tracker.update(dets[idx])
            self._add_tracker(new_tracker)

        for idx in unmatched_trackers:
            # print('unmat trk : %s' % (idx))
            self.trackers[idx].loose()

        self._discard_failed_tracker()

        boxes = []
        for t in self.trackers:
            if t.num_hits > 0:
                boxes.append(t.predict())
        return boxes
