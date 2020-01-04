import time
import datetime
from typing import Callable

import cv2
import numpy as np


FrameProcessor = Callable[[np.ndarray], np.ndarray]


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0
 
    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self
 
    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()
 
    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
 
    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()
 
    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


def webcam(process_frame_fn: FrameProcessor = None):
    if process_frame_fn is None:
        process_frame_fn = lambda x: x

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Unable to open camera")
        exit(-1)
    
    # Frame rate variables
    fps = 0
    frame_counter = 0
    prev_timer = time.time()

    while True:
        res, img = cap.read()
        if not res:
            continue
        
        img = process_frame_fn(img)

        # Compute Frame Rate
        frame_counter += 1
        now = time.time()
        if now - prev_timer >= 1:
            elapsed = now - prev_timer
            fps = frame_counter / elapsed
            prev_timer = time.time()
            frame_counter = 0

        x, y = 10, img.shape[0] - 10
        cv2.putText(img, f'FPS: {int(fps)}', (x, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, .5, 
                    (255,255,255), 2, cv2.LINE_AA)
        
        cv2.imshow('', img)
        k = cv2.waitKey(1)
        if k == 27:
            return