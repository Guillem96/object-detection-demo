import json
from pathlib import Path
from typing import Tuple, List

import click

import torch

from PIL import Image

import odet.utils.ss as ss
import odet.utils.bb as bb_utils
from odet.utils.typing import DetectionInstance


def _read_annot(annots_path: Path, 
                annot: Path) -> Tuple['Image', List[DetectionInstance]]:
    def shape_to_instance(shape):
        points = sum(shape['points'], [])
        return DetectionInstance(label=shape['label'], box=points)

    with annot.open() as f:
        annot_data = json.load(f)
    
    # Load the image
    image_path = str(annots_path / annot_data['imagePath'])
    
    # Load detection instances
    shapes = annot_data['shapes']
    instances = [shape_to_instance(s) for s in shapes]

    return image_path, instances


@click.command()
@click.option('--path', type=click.Path(file_okay=False, exists=True))
@click.option('--out', type=click.Path(file_okay=False))
def main(**args):
    out = Path(args['out'])
    out.mkdir(parents=True, exist_ok=True)

    path = Path(args['path'])
    for annot in path.glob('*.json'):
        image_path, instances = _read_annot(path, annot)
        
        im = Image.open(image_path)
        labels, boxes = zip(*instances)
        
        boxes = torch.FloatTensor(boxes)
        boxes = bb_utils.normalize_bbs(boxes, im.size[::-1])
        boxes = bb_utils.denormalize_bbs(boxes, (224, 224))

        im = im.resize((224, 224))
        search_regions = ss.ss(im, normalize_bbs=False)

        ious = bb_utils.overlap(search_regions, boxes)
        search_region_box = ious.argmax(1)
        max_overlaps = ious[torch.arange(search_region_box.size(0)), 
                            search_region_box]
        positive_indices = max_overlaps >= .5
        negative_indices = max_overlaps <= .4

        regression_targets = \
            bb_utils.bbox_transform(search_regions.float(), 
                                    boxes[search_region_box])
        # Filter the ignored indices
        regression_targets = torch.cat(
            [regression_targets[positive_indices],
             regression_targets[negative_indices]], dim=0).tolist()

        true_boxes = search_regions[positive_indices].tolist()
        false_boxes = search_regions[negative_indices].tolist()

        labels = [labels[i.item()] for i in search_region_box[positive_indices]]
        labels.extend(['background'] * len(false_boxes))
        boxes = boxes.tolist()

        instance = dict(
            image_path=image_path,
            true_regions=true_boxes,
            false_regions=false_boxes,
            regression_targets=regression_targets,
            labels=labels,
            gt_boxes=boxes
        )

        out_f = out / (annot.stem + '.json')
        with out_f.open('w') as f:
            json.dump(instance, f)


if __name__ == "__main__":
    main()

