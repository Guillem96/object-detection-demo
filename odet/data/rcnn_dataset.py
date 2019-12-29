import json
from pathlib import Path
from typing import List, Union, Tuple

import torch
from PIL import Image

import odet.utils.bb as bb_utils
from odet.utils.typing import DetectionInstance


class RCNNDataset(torch.utils.data.Dataset):

    def __init__(self,
                 annotation_path: Union[str, Path],
                 classes = List[str],
                 transforms = None):

        super(RCNNDataset, self).__init__()
        self.classes = ['background'] + classes
        self.label2idx = {c: i for i, c in enumerate(self.classes)}
        
        self.annots_path = Path(annotation_path)
        self.annots = list(self.annots_path.glob('*.json'))
        
        self.transforms = transforms

    def _read_annot(self, idx: int) -> Tuple['Image', List[DetectionInstance]]:
        def shape_to_instance(shape):
            points = sum(shape['points'], [])
            return DetectionInstance(label=shape['label'], box=points)

        with self.annots[idx].open() as f:
            annot_data = json.load(f)
        
        # Load the image
        image_path = annot_data['image_path']
        # image_path = '/gdrive/My Drive/_packages/object-detection-demo/' + image_path

        im = Image.open(image_path)
        
        return (im,
                annot_data['gt_boxes'], 
                annot_data['true_regions'],
                annot_data['false_regions'],
                annot_data['regression_targets'],
                annot_data['labels'])

    def __getitem__(self, idx: Union[int, torch.LongTensor]):
        if isinstance(idx, torch.LongTensor):
            idx = idx.item()

        instance = self._read_annot(idx)
        im, gt_boxes, positive, negative, rt, labels = instance

        rt = torch.FloatTensor(rt)
        labels = torch.LongTensor([self.label2idx[l] for l in labels])
        boxes = torch.cat([torch.FloatTensor(positive), 
                           torch.FloatTensor(negative)], dim=0)
       
        # Remove nans
        keep_idx = torch.all(rt == rt, dim=1)
 
        boxes = boxes[keep_idx]
        labels = labels[keep_idx]
        rt = rt[keep_idx].clamp(-100, 100)

        annots = dict(
            gt_boxes=torch.FloatTensor(gt_boxes),
            labels=labels,
            regression_targets=rt,
            ss_boxes=boxes,
        )

        if self.transforms:
            im = self.transforms(im)
        
        return im, annots

    def __len__(self) -> int:
        return len(self.annots)

