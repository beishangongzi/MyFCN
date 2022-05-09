# create by andy at 2022/5/8
# reference:
from torch import nn


class FCN8(nn.Module):
    def forward(self, x):
        feature_32 = self.features(x)["pool32"]
        feature_16 = self.features(x)["pool16"]
        feature_8 = self.features(x)["pool8"]

        y = self.up_sampling(feature_32, feature_16, 512)
        y = self.up_sampling(y, feature_8, 256)
        y = self.up_sampling(y, None, 128)
        y = self.up_sampling(y, None, 64)
        y = self.up_sampling(y, None, 32)
        y = self.classifier(y)
        y = self.classifier(y)
        return y

if __name__ == '__main__':
    pass
