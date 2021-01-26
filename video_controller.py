import threading
from time import sleep
import cv2

class VideoController(object):
    def __init__(self, video_name, scale, window_name):
        self.cap = cv2.VideoCapture(video_name)
        self.on_frame_available = lambda frame: frame
        self.is_pause = False
        self.is_stop = False
        self.lock = threading.Lock()
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.scale = scale
        cv2.createTrackbar('time', window_name, 0, self.n_frames, self.on_change)
        self.capture_thread = threading.Thread(target=self._capture)
        self.capture_thread.start()

    def _capture(self):
        while not self.is_stop:
            if self.is_pause:
                sleep(0.01)
                continue
            self.lock.acquire()
            ret, frame = self.cap.read()
            self.lock.release()
            if not ret:
                break
            h, w, _ = frame.shape
            frame = cv2.resize(frame, (int(w * self.scale), int(h * self.scale)))
            self.on_frame_available(frame)
        self.is_stop = True

    def set_on_frame_available(self, callback):
        self.on_frame_available = callback

    def on_change(self, pos):
        self.lock.acquire()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self.lock.release()

    def previous_step(self, sec=1):
        self.lock.acquire()
        cur_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        cur_frame = max(int(cur_frame - self.fps *sec), 0)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        self.lock.release()
        return cur_frame

    def next_step(self, sec=1):
        self.lock.acquire()
        cur_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        cur_frame = min(int(cur_frame + self.fps * sec), self.n_frames-1)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        self.lock.release()
        return cur_frame

    def stop(self):
        self.is_stop = True
        self.capture_thread.join()
        self.cap.release()

    def pause(self):
        self.is_pause = True

    def play(self):
        self.is_pause = False
