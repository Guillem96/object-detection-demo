from typing import Tuple

import torch
import numpy as np

import cv2
from PIL import Image


def _to_coordinates(rects: np.ndarray) -> np.ndarray:
    x, y, w, h = np.split(rects, 4, axis=1)
    return np.c_[x, y, x + w, y + h]


def _normalize_rects(rects: np.ndarray, 
                     image_size: Tuple[int, int]) -> np.ndarray:
    x1, y1, x2, y2 = np.split(rects, 4, axis=1)

    h, w = image_size

    x1 = x1 / (w - 1)
    x2 = x2 / (w - 1)
    y1 = y1 / (h - 1)
    y2 = y2 / (h - 1)
    return np.c_[x1, y1, x2, y2]


def ss(im: Image, 
       normalize_bbs: bool = True) -> torch.FloatTensor:
    im = np.array(im)[:, :, ::-1] # Image to BGR
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rects = _to_coordinates(rects)

    if normalize_bbs:
        rects = _normalize_rects(rects, im.shape[:2])
    
    return torch.from_numpy(rects)
