# create by andy at 2022/5/8
# reference:
from torch import nn


class FCN8(nn.Module):
    def forward(self, x):
        feature_32 = self.features(x)["pool32"]
        feature_16 = self.features(x)["pool16"]
        feature_8 = self.features(x)["pool8"]

        y = self.deconv1(feature_32)
        y = self.relu(y)
        y = self.bn1(y + feature_16)

        y = self.deconv2(y)
        y = self.relu(y)
        y = self.bn2(y + feature_8)

        y = self.deconv3(y)
        y = self.relu(y)
        y = self.bn3(y)

        y = self.deconv4(y)
        y = self.relu(y)
        y = self.bn4(y)

        y = self.deconv5(y)
        y = self.relu(y)
        y = self.bn5(y)
        y = self.classifier(y)
        return y

if __name__ == '__main__':
    pass
