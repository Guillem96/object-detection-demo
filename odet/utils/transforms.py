import random
from typing import List

import torch

from PIL import Image, ImageFilter, ImageOps


class UnNormalize(object):
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.FloatTensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class RandomBlur(object):

    def __init__(self, p: float = .5):
        self.p = p

    def __call__(self, image: 'Image') -> 'Image':
        if random.random() < self.p:
            image = image.filter(ImageFilter.BLUR)
        return image


class RandomSolarize(object):

    def __init__(self, intensity: int = 128, 
                 p: float = .5):
        self.intensity = intensity
        self.p = p

    def __call__(self, image: 'Image') -> 'Image':
        if random.random() < self.p:
            image = ImageOps.solarize(image, self.intensity)
        return image