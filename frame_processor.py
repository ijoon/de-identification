import copy
import os
import time
import threading

import cv2
import numpy as np

from face_register import FaceRegister
import fake_detector
from simple_tracker import TrackerPool, calc_iou
from utils import draw_rect, make_low_quality, get_closest_box_arg


FACENET = 'models/MobileNetV3_adam_relu6_gavg_arcface_60_trillion.h5'


class FrameProcessor(object):

    def __init__(self, window_name):
        self.selected_pos_idx = None
        self.selected_neg_idx = None
        self.current_img = None
        self.result_img = None
        self.face_boxes = None
        self.license_boxes = None
        self.window_name = window_name
        self.face_register = FaceRegister(model_file=FACENET)
        self.lock = threading.Lock()
        self.face_tracker = TrackerPool(num_tracker=5, iou_thr=0.2, patience=7)
        self.face_detector = fake_detector.FakeFaceDet()
        cv2.setMouseCallback(window_name, self._mouse_callback)

    def process(self, img):
        """
        이미지를 입력받아 처리합니다.
        이 함수는 비디오 클래스에서 이미지가 사용 가능할 때 콜백됩니다.
        """
        self.selected_pos_idx = None
        self.selected_neg_idx = None
        self.current_img = copy.deepcopy(img)
        bbox_dict = self.detect_objects(img)
        self.face_boxes = bbox_dict[0]
        self.license_boxes = bbox_dict[1]
        self.process_face(img, self.face_boxes)
        self.process_license(img, self.license_boxes)
        self.result_img = img

    def process_face(self, img, bboxes):
        """
        얼굴 영역을 처리합니다.
        """
        classifed_faces = self.face_register.classify(img, bboxes)
        if not self.face_register.clf.is_trained:
            for b in bboxes:
                cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
        else:
            target_boxes = self.face_tracker.assign(classifed_faces['target'])
            for box in classifed_faces['none']:
                for tbox in target_boxes:
                    if calc_iou(box, tbox) < 0.4:
                        make_low_quality(img, box)
                if len(target_boxes) == 0:
                    make_low_quality(img, box)

    def process_license(self, img, bboxes):
        """
        번호판 영역을 처리합니다.
        """
        for box in bboxes:
            make_low_quality(img, box)

    def detect_objects(self, img):
        """
        얼굴과 번호판을 검출하여 좌표를 반환합니다.
        입력: RGB 이미지
        출력: dictionary
        출력 dictionary 객체에 0번 키에는 얼굴 검출 박스.
        출력 dictionary 객체에 1번 키에는 번호판 검출 박스.
        """
        detected = {}

        """
        여기에서 얼굴 검출을 해주세요.
        """
        face_bboxes = self.face_detector.detect(img)
        detected[0] =  face_bboxes

        """
        여기에서 번호판 검출을 해주세요.
        """
        # license_bboxes = license_detector(img)
        # detected[1] = license_bboxes
        detected[1] = []

        return detected

    def get_processed_image(self):
        return self.result_img

    def _mouse_callback(self, event, x,y, flags, param):
        """
        마우스 클릭을 통해 얼굴을 등록할 수 있는 기능입니다.
        왼쪽 버튼은 등록할 얼굴을 지정합니다.
        오른쪽 버튼은 오인식 얼굴을 제거하기 위해 사용합니다.
        왼쪽 버튼이 클릭되면 해당 bounding box의 index가 self.selected_pos_idx 에 저장됩니다.
        오른쪽 버튼이 클릭되면 해당 bounding box의 index가 self.selected_neg_idx 에 저장됩니다.
        """
        
        if event == cv2.EVENT_LBUTTONUP:
            img = copy.deepcopy(self.current_img)
            closest_bbox_idx = get_closest_box_arg(x, y, self.face_boxes)
            if closest_bbox_idx < 0:
                return
            self.selected_pos_idx = closest_bbox_idx
            b = self.face_boxes[closest_bbox_idx]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 255, 255), 3)
            self.result_img = img
        elif event == cv2.EVENT_RBUTTONUP:
            img = copy.deepcopy(self.current_img)
            closest_bbox_idx = get_closest_box_arg(x, y, self.face_boxes)
            if closest_bbox_idx < 0:
                return
            self.selected_neg_idx = closest_bbox_idx
            b = self.face_boxes[closest_bbox_idx]
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
            self.result_img = img

    def save_reference_faces(self, save_folder):
        target_dir = '{}/0'.format(save_folder)
        other_dir = '{}/1'.format(save_folder)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        if not os.path.exists(other_dir):
            os.mkdir(other_dir)
        if self.selected_pos_idx is not None:
            for i, b in enumerate(self.face_boxes):
                b = np.array(b).astype(np.int32)
                img = self.current_img[b[1]:b[3], b[0]:b[2]]
                uniq_id = int(time.time())
                if self.selected_pos_idx == i:
                    cv2.imwrite('{}/{}.jpg'.format(target_dir, uniq_id), img)
                else:
                    cv2.imwrite('{}/{}_{}.jpg'.format(other_dir, uniq_id, i), img)
        elif self.selected_neg_idx is not None:
            b = self.face_boxes[self.selected_neg_idx]
            b = np.array(b).astype(np.int32)
            img = self.current_img[b[1]:b[3], b[0]:b[2]]
            uniq_id = int(time.time())
            cv2.imwrite('{}/{}_{}.jpg'.format(other_dir, uniq_id, 0), img)
