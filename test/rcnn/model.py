import unittest

import torch

import odet.models.rcnn as rcnn
import test.utils as test_utils


class RCNNModelsTest(unittest.TestCase):
    
    def test_feature_extractor_forward(self):
        fe = rcnn.FeatureExtractor()
        x = torch.rand(3, 3, 224, 224)
        out = fe(x)
        test_utils.assert_shapes([None, 2048, 1, 1], out.shape)

    def test_classifier_predict(self):
        regions = torch.ones(10, 3, 224, 224)
        clf = rcnn.Classifier(5)
        clf.fit(regions, torch.randint(high=5, size=(10,)))

        probas = clf.predict(regions)
        print(probas)

if __name__ == "__main__":
    unittest.main()