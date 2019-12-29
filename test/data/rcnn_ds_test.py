import random
import unittest

import odet.data as data_utils
import odet.utils.bb as bb_utils
import torchvision.transforms as T


class RCNNDsTest(unittest.TestCase):

    def test_labelme_ds(self):
        classes = ['treecko', 'mewtwo', 'solgaleo', 'psyduck', 'greninja']
        ds = data_utils.RCNNDataset(
            annotation_path='data/pokemon/ss',
            classes=classes,
            transforms=T.Resize((224, 224)))
        
        idx = random.randint(0, len(ds) - 1)
        im, annots = ds[idx]

        true_mask = annots['labels'] != 0

        ss_boxes = annots['ss_boxes']
        true_boxes = ss_boxes[true_mask]

        # Apply regression
        regress_targets = annots['regression_targets'][true_mask]
        true_boxes = bb_utils.regress_bndboxes(true_boxes, regress_targets)
        print(true_boxes)
        labels = [classes[i.item() - 1] for i in annots['labels'][true_mask]]
        
        bb_utils.draw_bbs(im, true_boxes, labels=labels) 

        im.show()


if __name__ == "__main__":
    unittest.main()