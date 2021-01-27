import cv2
import numpy as np

def draw_rect(img, box, color, thickness=2):
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color, thickness)

def make_low_quality(img, area):
    """
    주어진 영역의 이미지 퀄리티를 낮춥니다.
    """
    area = np.array(area, dtype=np.int32)
    crop = img[area[1]:area[3], area[0]:area[2]]
    shape = crop.shape[:2]
    crop = cv2.resize(crop, (40, 40))
    crop = cv2.GaussianBlur(crop, (7, 7), 5)
    crop = cv2.resize(crop, None, fx=0.05, fy=0.05)
    crop = cv2.resize(crop, (shape[1], shape[0]))
    img[area[1]:area[3], area[0]:area[2]] = crop

def get_closest_box_arg(x, y, bboxes):
    """
    주어진 좌표와 가장 가까운 bounding box의 index를 반환합니다.
    """
    closest_bbox_idx = -1
    min_distance = 1e+10
    for i, b in enumerate(bboxes):
        if (x > b[0] and x < b[2]) and (y > b[1] and y < b[3]):
            cx = (b[2] - b[0]) / 2
            cy = (b[3] - b[1]) / 2
            dx = np.abs(cx-x)
            dy = np.abs(cy-y)
            distance = np.sqrt(dx**2 + dy**2)
            if distance < min_distance:
                min_distance = distance
                closest_bbox_idx = i
    return closest_bbox_idx
