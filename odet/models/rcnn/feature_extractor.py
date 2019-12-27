import torch
import torch.nn as nn

import torchvision.models as zoo


class FeatureExtractor(nn.Module):

    def __init__(self, freeze: bool = True):
        super(FeatureExtractor, self).__init__()
        
        resnet = zoo.resnet50(pretrained=True)
        self.fe = nn.Sequential(
            *[m for n, m in resnet.named_children() if n != 'fc'])
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.fe(x)


if __name__ == "__main__":
    fe = FeatureExtractor()
    im = torch.randn(10, 3, 224, 224)
    print(fe(im).shape)