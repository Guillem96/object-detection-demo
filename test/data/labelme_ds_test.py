import random
import unittest

import torch
from PIL import Image
import albumentations as albu

import odet.data as data_utils
import odet.utils.bb as bb_utils
import torchvision.transforms as T


class LabelMeDsTest(unittest.TestCase):

    def test_labelme_ds(self):
        classes = ['treecko', 'mewtwo', 'solgaleo', 'psyduck', 'greninja']
        ds = data_utils.LabelMeDataset(
            root='data/pokemon',
            images_path='data/pokemon',
            classes=classes)
        
        annots = ds[random.randint(0, len(ds))]
        im = Image.fromarray(annots['image'][:,:,::-1])
        im = im.resize((512, 512))

        boxes = bb_utils.scale_bbs(annots['bboxes'], 
                                   (3456, 3456), im.size)
        labels = [classes[i.item()] for i in annots['category_id']]
        bb_utils.draw_bbs(im, boxes, labels=labels) 

        im.show()


if __name__ == "__main__":
    unittest.main()