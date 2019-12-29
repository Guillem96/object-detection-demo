from typing import Tuple

import torch
import numpy as np

import cv2
from PIL import Image

import odet.utils.bb as bb_utils


def _to_coordinates(rects: np.ndarray) -> np.ndarray:
    x, y, w, h = np.split(rects, 4, axis=1)
    return np.c_[x, y, x + w, y + h]


def ss(im: Image, 
       normalize_bbs: bool = True) -> torch.FloatTensor:
    im = np.array(im)[:, :, ::-1] # Image to BGR
    
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(im)
    ss.switchToSelectiveSearchFast()
    rects = ss.process()
    rects = _to_coordinates(rects)

    if normalize_bbs:
        return bb_utils.normalize_bbs(torch.from_numpy(rects).float(), 
                                      im.shape[:2])
    
    return torch.from_numpy(rects)
