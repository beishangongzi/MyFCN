# create by andy at 2022/5/8
# reference:
import torch


from model.FCN import FCN


class FCN32(FCN):
    def forward(self, x):
        feature = self.features(x)["pool32"]
        y = self.deconv1(feature)
        y = self.relu(y)
        y = self.bn1(y)

        y = self.deconv2(y)
        y = self.relu(y)
        y = self.bn2(y)

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

def test():
    test_data = torch.zeros([1, 32, 256, 256])
    model = FCN32(5, "vgg16")
    model(test_data)

if __name__ == '__main__':
    test()
