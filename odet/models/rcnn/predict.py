import functools
from typing import List

import click

import cv2
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

import numpy as np

import odet.utils.ss as ss
import odet.models.rcnn as rcnn
import odet.utils.camera as cam
import odet.utils.bb as bb_utils
import odet.utils.transforms as odet_T

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tfms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def _draw_boxes_cv(im: np.ndarray, 
                   boxes: torch.FloatTensor,
                   labels: List[str]):
    boxes = boxes.int().tolist()
    colors = {
        'treecko': (64, 199, 109),
        'mewtwo': (207, 129, 219),
        'greninja': (94, 143, 204),
        'psyduck': (222, 224, 79),
        'solgaleo': (255, 255, 255)
    }
    for (x1, y1, x2, y2), l in zip(boxes, labels):
        cv2.rectangle(im, (x1, y1), (x2, y2), colors[l][::-1], 2)
    
    return im


def crop_and_resize(im: 'Image', 
                    boxes: torch.FloatTensor) -> torch.FloatTensor:
    def crop_single(box):
        crop = im[:, box[1]: box[3], box[0]: box[2]]
        return F.interpolate(crop.unsqueeze(0), size=(224, 224))

    return torch.cat([crop_single(b) for b in boxes], dim=0)


@torch.no_grad()
def _predict_video(im: np.ndarray,
                   model: torch.nn.Module,
                   classes: List[str]) -> np.ndarray:
    n_regions = 1024
    im_input = cv2.resize(im, (224, 224))
    regions = ss.ss(im_input, normalize_bbs=False)[:n_regions]

    im_input = np.flip(im_input, axis=-1).copy()
    im_input = torch.from_numpy(im_input).permute(2, 0, 1)
    im_input = im_input.float().div(255.)
    im_input = T.functional.normalize(im_input, 
                                      mean=[0.485, 0.456, 0.406], 
                                      std=[0.229, 0.224, 0.225])

    crops = crop_and_resize(im_input, regions)
    clf_logits = []
    reg_targets = []
    # Feed with batches of 32 regions
    for i in range(0, n_regions, 32):
        cc = crops[i: i + 32]
        c_clf, c_reg = model(cc.to(DEVICE))
        clf_logits.append(c_clf)
        reg_targets.append(c_reg)

    scores = torch.cat(clf_logits, dim=0).softmax(-1)
    boxes = bb_utils.regress_bndboxes(regions.to(DEVICE), torch.cat(reg_targets, dim=0))
    boxes, labels = bb_utils.nms(boxes, scores)
    
    im = cv2.resize(im, (512, 512))
    if boxes is None:
        return im

    labels = [classes[i.item() - 1] for i in labels]
    boxes = bb_utils.scale_bbs(boxes, (224, 224), im.shape)

    return _draw_boxes_cv(im, boxes, labels)


@torch.no_grad()
def _predict_single_image(model: torch.nn.Module, im: 'Image'):
    im = im.resize((224, 224))
    regions = ss.ss(im, normalize_bbs=False)[:128]
    
    input_im = tfms(im)
    crops = crop_and_resize(input_im, regions).to(DEVICE)
    
    clf_logits, reg_targets = model(crops)

    scores = clf_logits.softmax(-1)

    boxes = bb_utils.regress_bndboxes(regions, reg_targets)
    boxes, labels = bb_utils.nms(boxes, scores)
    
    return boxes, labels


@click.command()
@click.option('--image', 
              type=click.Path(exists=True, dir_okay=False))
@click.option('--checkpoint',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--video/--no-video', default=False)
def main(**args):
    classes = ['treecko', 'mewtwo', 'greninja', 'psyduck', 'solgaleo']

    model = rcnn.RCNN(n_classes=6)
    model.to(DEVICE)
    model.eval()

    chkp = torch.load(args['checkpoint'], map_location=DEVICE)
    model.load_state_dict(chkp)

    if args['image']:
        im = Image.open(args['image'])
        boxes, labels = _predict_single_image(model, im)
        labels = [classes[i.item() - 1] for i in labels]

        # Prepare boxes for display
        im = im.resize((512, 512))
        boxes = bb_utils.scale_bbs(boxes, (224, 224), im.size)
        bb_utils.draw_bbs(im, boxes, labels)
        
        im.show()

    if args['video']:
        print('Starting video..')
        process_frame_fn = functools.partial(_predict_video, 
                                             model=model, 
                                             classes=classes)
        cam.webcam(process_frame_fn)


if __name__ == "__main__":
    main()
