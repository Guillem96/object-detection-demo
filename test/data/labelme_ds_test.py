import random
import unittest

import odet.data as data_utils
import odet.utils.bb as bb_utils
import torchvision.transforms as T


class LabelMeDsTest(unittest.TestCase):

    def test_labelme_ds(self):
        classes = ['treecko', 'mewtwo', 'solgaleo', 'psyduck', 'greninja']
        ds = data_utils.LabelMeDataset(
            annotation_path='data/pokemon',
            classes=classes,
            transforms=T.Resize((224, 224)))
        
        im, annots = ds[random.randint(0, len(ds))]

        im = im.resize((224, 224))
        bbs = bb_utils.denormalize_bbs(annots['boxes'], im.size[::-1])
        labels = [classes[i.item() - 1] for i in annots['labels']]
        bb_utils.draw_bbs(im, bbs, labels=labels) 

        im.show()


if __name__ == "__main__":
    unittest.main()