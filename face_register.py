from face_module import FaceModule
from classifier import Calssifier

import cv2
import numpy as np

class FaceRegister(object):

    def __init__(self, model_file):
        self.fm = FaceModule(model_file)
        self.clf = Calssifier()

    def classify(self, img, bboxes):
        faces = []
        for b in bboxes:
            b = np.array(b, dtype=np.int32)
            if np.amin(b) < 0:
                continue
            crop = img[b[1]:b[3], b[0]:b[2]]
            crop = cv2.resize(crop, (112, 112))
            faces.append(crop)
        classified = {
            'target': [],
            'none': []
        }
        if self.clf.is_trained and len(faces) != 0:
            faces = np.array(faces)
            embedding = self.fm(faces)
            labels = self.clf.predict(embedding)
            for label, b in zip(labels, bboxes):
                if label == 0:
                    classified['target'].append(b)
                else:
                    classified['none'].append(b)
        return classified

    def update(self, reference_folder):
        x, y = self.fm.get_train_data_from_dir(reference_folder)
        self.clf.fit(x, y)
