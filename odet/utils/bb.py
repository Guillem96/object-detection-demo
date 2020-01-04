import random
from typing import Tuple, List

import torch
import torchvision.ops as ops

from PIL import Image, ImageDraw

from .compute_overlap import compute_overlap


def normalize_bbs(boxes: torch.FloatTensor, 
                  im_size: Tuple[int, int]) -> torch.FloatTensor:

    x1, y1, x2, y2 = boxes.split(1, dim=1)

    h, w = im_size

    x1 = x1 / (w - 1)
    x2 = x2 / (w - 1)
    y1 = y1 / (h - 1)
    y2 = y2 / (h - 1)
    return torch.cat([x1, y1, x2, y2], dim=1)


def denormalize_bbs(boxes: torch.FloatTensor, 
                    im_size: Tuple[int, int]) -> torch.FloatTensor:

    x1, y1, x2, y2 = boxes.split(1, dim=1)

    h, w = im_size

    x1 = x1 * (w - 1)
    x2 = x2 * (w - 1)
    y1 = y1 * (h - 1)
    y2 = y2 * (h - 1)
    return torch.cat([x1, y1, x2, y2], dim=1)


def scale_bbs(boxes: torch.FloatTensor,
              boxes_im_size: Tuple[int, int],
              dest_im_size: Tuple[int, int]) -> torch.FloatTensor:
    
    factor_w = dest_im_size[0] / float(boxes_im_size[0])
    factor_h = dest_im_size[1] / float(boxes_im_size[1])

    x1, y1, x2, y2 = boxes.split(1, dim=1)

    x1 *= factor_w
    x2 *= factor_w
    y1 *= factor_h
    y2 *= factor_h

    return torch.cat([x1, y1, x2, y2], dim=1)

    
def _gen_random_colors(length: int) -> List[Tuple[int]]:
    def rand_color():
        return tuple(random.randint(0, 256) for _ in range(3))
    return [rand_color() for _ in range(length)]


def draw_bbs(im: Image, 
             boxes: torch.FloatTensor,
             labels: List[str],
             colors = []) -> 'Image':

    colors = colors or _gen_random_colors(len(set(labels)))
    colors = {l: c for c, l in zip(colors, set(labels))}

    draw = ImageDraw.Draw(im)
    for l, r in zip(labels, boxes.tolist()):
        draw.text((r[0], r[1] - 10), l, fill=colors[l])
        draw.rectangle(r, outline=colors[l], width=2)
    
    return im


def overlap(gt_boxes: torch.FloatTensor, 
            search_boxes: torch.FloatTensor) -> torch.FloatTensor:

    gt_boxes = gt_boxes.double()
    search_boxes = search_boxes.double()
    
    ious = compute_overlap(gt_boxes.numpy(), 
                           search_boxes.numpy())
    return torch.from_numpy(ious)


def to_rcnn_style(boxes):
    w = boxes[:, 2] - boxes[:, 0] + 1e-6
    h = boxes[:, 3] - boxes[:, 1] + 1e-6
    
    x = (boxes[:, 0] + boxes[:, 2]) / 2
    y = (boxes[:, 1] + boxes[:, 3]) / 2
    
    return x, y, w, h


def bbox_transform(search_boxes: torch.FloatTensor, 
                   gt_boxes: torch.FloatTensor,
                   normalize: bool = True) -> torch.FloatTensor:
    """Compute bounding-box regression targets for an image."""

    Px, Py, Pw, Ph = to_rcnn_style(search_boxes)
    Gx, Gy, Gw, Gh = to_rcnn_style(gt_boxes)

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = torch.log(Gw / Pw)
    th = torch.log(Gh / Ph)

    t = torch.stack([tx, ty, tw, th]).t()

    if normalize:
        mean = torch.zeros(4)
        std = torch.tensor([.2] * 4)

        return (t - mean) / std
    
    return t


def regress_bndboxes(boxes: torch.FloatTensor,
                     regressors: torch.FloatTensor,
                     from_normalized: bool = True) -> torch.FloatTensor:
    """
    Apply scale invariant regression to boxes.
    Parameters
    ----------
    boxes: torch.FloatTensor of shape [N_BOXES, 4]
        Boxes to apply the regressors
    regressors: torch.FloatTensor  of shape [N_BOXES, 4]
        Scale invariant regressions
    
    Returns
    -------
    torch.FloatTensor
        Regressed boxes
    """ 
    Px, Py, Pw, Ph = to_rcnn_style(boxes)
    split = regressors.split(1, dim=1)
    dx, dy, dw, dh = [s.squeeze(-1) for s in split]
    
    if from_normalized:
        mean = torch.zeros(4)
        std = torch.tensor([.2] * 4)
        dx = dx * std[0] + mean[0]
        dy = dy * std[1] + mean[1]
        dw = dw * std[2] + mean[2]
        dh = dh * std[3] + mean[3] 
    
    Gx = Pw * dx + Px
    Gy = Ph * dy + Py
    Gw = Pw * torch.exp(dw)
    Gh = Ph * torch.exp(dh)

    x = Gx - Gw / 2
    y = Gy - Gh / 2
    x2 = x + Gw
    y2 = y + Gh

    return torch.stack([x, y, x2, y2]).t() 


def nms(boxes: torch.FloatTensor, 
        class_scores: torch.FloatTensor) -> torch.FloatTensor:

    """
    Parameters
    ----------
    boxes: torch.FloatTensor of shape [N, 4]
    class_scores: torch.FloatTensor of shape [N, NUM_CLASSES]
    """
    score_threshold = .2
    iou_threshold = .5

    all_labels = []
    all_boxes = []

    for c in range(1, class_scores.size(-1)):
        scores = class_scores[:, c]
        best_boxes = boxes[scores > score_threshold]
        keep_idx = ops.nms(best_boxes, 
                           scores[scores > score_threshold], 
                           iou_threshold)
        all_labels.extend([c] * keep_idx.size(0))
        all_boxes.extend(best_boxes[keep_idx])
    
    return torch.stack(all_boxes), torch.LongTensor(all_labels)