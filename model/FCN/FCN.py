# create by andy at 2022/5/8
# reference:
from torch import nn

from model.FCN.vgg import VGG16


class FCN(nn.Module):
    def __init__(self, input_size, num_classes, backbone="vgg16"):
        super().__init__()
        all_backones = ["vgg16"]
        if backbone not in all_backones:
            raise ValueError(f"backbone must be ont of the item in {all_backones}")

        if backbone == "vgg16":
            self.features = VGG16(input_size)
            self.num_classes = num_classes

            self.deconv1 = nn.ConvTranspose2d(512, 512, 3, 2, padding=1, output_padding=1)
            self.bn1 = nn.BatchNorm2d(512)
            self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, padding=1, output_padding=1)
            self.bn2 = nn.BatchNorm2d(256)
            self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1)
            self.bn3 = nn.BatchNorm2d(128)
            self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1)
            self.bn4 = nn.BatchNorm2d(64)
            self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, padding=1, output_padding=1)
            self.bn5 = nn.BatchNorm2d(32)
            self.classifier = nn.Conv2d(32, num_classes, kernel_size=1, padding="same")
            self.bn = nn.BatchNorm2d
            self.relu = nn.ReLU()

    def forward(self, x):
        raise NotImplementedError("please implement it")

    def up_sampling(self, x1, x2=None, batch_norm=None):
        deconv = None
        assert batch_norm is not None
        if batch_norm == 512:
            deconv = self.deconv1
        elif batch_norm == 256:
            deconv = self.deconv2
        elif batch_norm == 128:
            deconv = self.deconv3
        elif batch_norm == 64:
            deconv = self.deconv4
        elif batch_norm == 32:
            deconv = self.deconv5
        y = deconv(x1)
        y = self.relu(y)
        if x2 is None:
            y = self.bn(batch_norm)(y)
        else:
            y = self.bn(batch_norm)(y + x2)
        return y


if __name__ == '__main__':
    pass
