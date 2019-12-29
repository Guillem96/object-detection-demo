import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as zoo


class FeatureExtractor(nn.Module):

    def __init__(self, freeze: bool = True):
        super(FeatureExtractor, self).__init__()
        
        resnet = zoo.resnet50(pretrained=True)
        self.fe = nn.Sequential(
            *[m for n, m in resnet.named_children() if n != 'fc'])
        
        if freeze:
            for p in resnet.parameters():
                p.requires_grad = False
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.fe(x)


class Regressor(nn.Module):
    def __init__(self, 
                 feature_size: int = 2048):
        
        super(Regressor, self).__init__()
        self.linear1 = nn.Linear(feature_size, 1024)
        self.dropout = nn.Dropout(.5)
        self.linear2 = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x)


class Classifier(nn.Module):

    def __init__(self, 
                 feature_size: int = 2048,
                 n_classes: int = 5):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(feature_size, 1024)
        self.dropout = nn.Dropout(.5)
        self.linear2 = nn.Linear(1024, n_classes)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear1(x)))
        return self.linear2(x)


class RCNN(nn.Module):
    
    def __init__(self, 
                 feature_extractor_freeze: bool = True,
                 n_classes: int = 5):
        super(RCNN, self).__init__()
        self.fe = FeatureExtractor(freeze=feature_extractor_freeze)
        self.clf = Classifier(n_classes=n_classes)
        self.regressor = Regressor()
        
    def forward(self, x):
        features = self.fe(x)
        features = features.view(x.size(0), -1)
        return self.clf(features), self.regressor(features)
    