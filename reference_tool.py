import argparse
import shutil
import os
from time import process_time, sleep
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from frame_processor import FrameProcessor
from video_controller import VideoController
import fake_detector

import cv2

WINDOW_NAME = 'de-identification tool'

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str,
    help='video file or camera device number')
parser.add_argument('--scale', type=float, default=1.0,
    help='frame rescale to detect')
args = parser.parse_args()

fake_detector.FAKE_FACE_FILE = args.file + '.json'
fake_detector.scale = args.scale

if __name__ == '__main__':
    save_folder = 'reference'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    fp = FrameProcessor(window_name=WINDOW_NAME)
    cap = VideoController(args.file, scale=args.scale, window_name=WINDOW_NAME)
    cap.set_on_frame_available(fp.process)
    cap.play()
    try:
        while not cap.is_stop:
            img = fp.get_processed_image()
            if img is not None:
                cv2.imshow(WINDOW_NAME, img)
            k = cv2.waitKey(1)
            if k == 27:
                break
            elif k == ord('t'):
                fp.face_register.update(save_folder)
            elif k == ord('s'):
                fp.save_reference_faces(save_folder)
            elif k == ord('j'):
                fake_detector.frame_count = cap.previous_step()
            elif k == ord('k'):
                fake_detector.frame_count = cap.next_step()
            elif k == 32:
                if cap.is_pause:
                    cap.play()
                else:
                    cap.pause()
    except KeyboardInterrupt as k:
        pass
    cap.stop()
