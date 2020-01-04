from typing import Tuple, Union

import torch
import numpy as np

import cv2
from PIL import Image

import odet.utils.bb as bb_utils


def _to_coordinates(rects: np.ndarray) -> np.ndarray:
    x, y, w, h = np.split(rects, 4, axis=1)
    return np.c_[x, y, x + w, y + h]


def ss(im: Union['Image', np.ndarray], 
       normalize_bbs: bool = True) -> torch.FloatTensor:
    
    if isinstance(im, np.ndarray):
        im = im[:, :, ::-1]
    else:
        im = np.array(im)[:, :, ::-1]
    
    ss_ = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss_.setBaseImage(im)
    ss_.switchToSelectiveSearchQuality()
    
    rects = ss_.process()
    rects = _to_coordinates(rects)

    if normalize_bbs:
        return bb_utils.normalize_bbs(torch.from_numpy(rects).float(), 
                                      im.shape[:2])
    
    return torch.from_numpy(rects)
