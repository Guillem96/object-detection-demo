import functools
from typing import List

import click

import cv2

import torch

import numpy as np

import odet.utils.camera as cam
from .augmentation import get_augmentation
import odet.models.efficientdet as efficientdet


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tfms = get_augmentation(phase='test')


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


@torch.no_grad()
def _predict_single_image(model: torch.nn.Module, 
                          im: np.ndarray):
    im_input = tfms(image=im)
    _, labels, boxes = model(im_input['image'].to(DEVICE).unsqueeze(0))
    return boxes, labels


@torch.no_grad()
def _predict_video(im: np.ndarray,
                   model: torch.nn.Module,
                   classes: List[str]) -> np.ndarray:
    im_input = tfms(image=im.copy()).to(DEVICE)
    _, labels, boxes = model(im_input['image'].unsqueeze(0))
    
    im = cv2.resize(im, (512, 512))
    labels = [classes[i.item()] for i in labels]

    return _draw_boxes_cv(im, boxes, labels)



@click.command()
@click.option('--image', 
              type=click.Path(exists=True, dir_okay=False))
@click.option('--checkpoint',
              type=click.Path(exists=True, dir_okay=False))
@click.option('--video/--no-video', default=False)
def main(**args):
    classes = ['treecko', 'mewtwo', 'greninja', 'psyduck', 'solgaleo']
    
    chkp = torch.load(args['checkpoint'], map_location=DEVICE)
    n_classes = chkp['num_class']
    net = chkp['network']
    net_opts = efficientdet.EFFICIENTDET[net]

    model = efficientdet.EfficientDet(num_classes=n_classes,
                                      network=net,
                                      W_bifpn=net_opts['W_bifpn'],
                                      D_bifpn=net_opts['D_bifpn'],
                                      D_class=net_opts['D_class'],
                                      is_training=False)
    model.to(DEVICE)
    model.eval()

    model.load_state_dict(chkp['state_dict'])

    if args['image']:
        im = cv2.imread(args['image'])
        
        boxes, labels = _predict_single_image(model, im.copy())
        labels = [classes[i.item()] for i in labels]

        # Prepare boxes for display
        im = cv2.resize(im, (512, 512))
        im = _draw_boxes_cv(im.copy(), boxes, labels)
        
        cv2.imshow('', im)
        cv2.waitKey(0)

    if args['video']:
        process_frame_fn = functools.partial(_predict_video, 
                                             model=model, 
                                             classes=classes)
        cam.webcam(process_frame_fn)


if __name__ == "__main__":
    main()