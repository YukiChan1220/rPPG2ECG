import cv2
import numpy as np
from queue import Queue
import time
import os
from .base import PreprocessBase
from typing import Tuple, Generator, Optional
import global_vars


class Video2Frame(PreprocessBase):
    def __init__(self):
        self.path = None
        self.video_path = None
        self.ts_path = None
        self.cap = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        self.processed_frames = 0
        global_vars.preprocess_completed = False

    def _preprocess_conv_format(self, frame) -> np.ndarray:
        frame = cv2.resize(frame, (36, 36))
        return frame#.astype("float32")

    def _load_timestamps(self):
        timestamps = []
        try:
            with open(self.ts_path, 'r') as f:
                lines = f.readlines()
                for line in lines[1:]:  # Skip header
                    parts = line.strip().split(', ')
                    if len(parts) == 2:
                        timestamps.append(float(parts[1]))
        except Exception as e:
            print(f"Error reading timestamp file: {e}")
            return []
        return timestamps

    def __call__(self, preprocess_queue: Queue):
        self.video_path = self.path + "/video.avi"
        self.ts_path = self.path + "/video.avi.ts"

        timestamps = self._load_timestamps()
        
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            self.cap.release()
            global_vars.preprocess_completed = True
            return
            raise ValueError(f"Error opening video file: {self.video_path}")

        

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video opened: {self.video_path}")
        print(f"Total frames: {self.total_frames}, Width: {self.frame_width}, Height: {self.frame_height}")
        print(f"Loaded {len(timestamps)} timestamps")

        self.processed_frames = 0
        while not global_vars.user_interrupt:
            ret, raw_frame = self.cap.read()
            if not ret:
                ret, raw_frame = self.cap.read()
                if not ret:
                    print("End of video or cannot read the frame.")
                    break

            timestamp = timestamps[self.processed_frames]
                
            preprocess_queue.put((self._preprocess_conv_format(raw_frame), timestamp))
            self.processed_frames += 1

        self.cap.release()
        global_vars.preprocess_completed = True
