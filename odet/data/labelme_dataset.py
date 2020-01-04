import json
from pathlib import Path
from typing import List, Union, Tuple

import torch
import cv2
from PIL import Image

import odet.utils.bb as bb_utils
from odet.utils.typing import DetectionInstance


class LabelMeDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root: Union[str, Path],
                 images_path: Union[str, Path],
                 classes = List[str],
                 transforms = None):

        super(LabelMeDataset, self).__init__()
        self.classes = classes
        self.label2idx = {c: i for i, c in enumerate(self.classes)}
        self.annots_path = Path(root)
        self.annots = list(self.annots_path.glob('*.json'))
        
        self.images_path = Path(images_path)
        
        self.transforms = transforms

    def _read_annot(self, idx: int) -> Tuple['Image', List[DetectionInstance]]:
        def shape_to_instance(shape):
            points = sum(shape['points'], [])
            return DetectionInstance(label=shape['label'], box=points)

        with self.annots[idx].open() as f:
            annot_data = json.load(f)
        
        # Load the image
        image_path = str(self.images_path / annot_data['imagePath'])
        im = cv2.imread(image_path)
        
        # Load detection instances
        shapes = annot_data['shapes']
        instances = [shape_to_instance(s) for s in shapes]

        return im, instances

    def __getitem__(self, idx: Union[int, torch.LongTensor]):
        if isinstance(idx, torch.LongTensor):
            idx = idx.item()

        im, boxes = self._read_annot(idx)
        labels, bbs = zip(*boxes)
        
        bbs = torch.FloatTensor(bbs)
        labels = torch.LongTensor([self.label2idx[l] for l in labels])
        
        instance = dict(image=im, bboxes=bbs, category_id=labels)
        if self.transforms is not None:
            instance = self.transforms(**instance)
        
        return instance

    def __len__(self) -> int:
        return len(self.annots)

