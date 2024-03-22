import torchvision.models as models
import torch.nn as nn

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        vgg19 = models.vgg19(pretrained=True).features

        self.feature_extractor = nn.Sequential()

        # Add layers up to 'pool4' (which is the 27th layer of features)
        for layer in range(28):  # 27 is the index of 'pool4' plus one to include 'pool4' itself
            self.feature_extractor.add_module(str(layer), vgg19[layer])

    def forward(self, x):
        return self.feature_extractor(x)
