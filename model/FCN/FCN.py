# create by andy at 2022/5/8
# reference:
from torch import nn

from model.FCN.ResNet import ResNet
from model.FCN.vgg import VGG16


class FCN(nn.Module):
    def __init__(self, input_size, num_classes, backbone="vgg16"):
        super().__init__()
        all_backones = ["vgg16", "resnet50", "resnet101", "resnet152"]
        if backbone == "vgg16":
            self.features = VGG16()
            self.num_classes = num_classes
            input_channels = 512
        elif backbone == "resnet50":
            self.features = ResNet.get_resnet50()
            input_channels = 2048
        elif backbone == "resnet101":
            self.features = ResNet.get_resnet101()
            input_channels = 2048
        elif backbone == "resnet152":
            self.features = ResNet.get_resnet152()
            input_channels = 2048
        else:
            raise ValueError(f"backbone must be ont of the item in {all_backones}")

        self.num_classes = num_classes
        self.deconv1 = nn.ConvTranspose2d(input_channels, input_channels, 3, 2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.deconv2 = nn.ConvTranspose2d(input_channels, input_channels//2, 3, 2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(input_channels//2)
        self.deconv3 = nn.ConvTranspose2d(input_channels//2, input_channels//4, 3, 2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(input_channels//4)
        self.deconv4 = nn.ConvTranspose2d(input_channels//4, input_channels//8, 3, 2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(input_channels//8)
        self.deconv5 = nn.ConvTranspose2d(input_channels//8, input_channels//16, 3, 2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(input_channels//16)
        self.classifier = nn.Conv2d(input_channels//16, num_classes, kernel_size=1, padding="same")
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
