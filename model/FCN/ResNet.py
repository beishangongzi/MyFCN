# create by andy at 2022/5/10
# reference: 
import torch
import torch.nn as nn
import torchvision
import numpy as np

print("Pytorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

__all__ = ["ResNet50", "ResNet101", "ResNet152"]


def Conv1(in_channels, out_channels, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=stride, padding=3,
                  bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )


class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sampling=False, expansion=4):
        super(BottleNeck, self).__init__()
        self.expansion = expansion
        self.down_sampling = down_sampling

        self.bottle_neck = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        if self.down_sampling:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.bottle_neck(x)
        if self.down_sampling:
            residual = self.down_sample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=1000, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        self.conv1 = Conv1(in_channels=3, out_channels=64)

        self.layer1 = self.make_layer(64, 64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(256, 128, blocks[1], 2)
        self.layer3 = self.make_layer(512, 256, blocks[2], 2)
        self.layer4 = self.make_layer(1024, 512, blocks[3], 2)

        self.avgpool = nn.AvgPool2d(7, 1)
        self.fc = nn.Linear(2048, num_classes)

    def make_layer(self, in_channels, out_channels, block, stride):
        layers = []
        layers.append(BottleNeck(in_channels, out_channels, stride, True))
        for i in range(1, block):
            layers.append(BottleNeck(out_channels * self.expansion, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        # 4s
        x = self.layer1(x)
        # 8s
        x = self.layer2(x)
        pool_8s = x
        # 16s
        x = self.layer3(x)
        pool_16s = x
        # 32s
        x = self.layer4(x)
        pool_32s = x

        return {"pool32": pool_32s,
                "pool16": pool_16s,
                "pool8": pool_8s}

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

    @staticmethod
    def get_resnet50():
        return ResNet([3, 4, 6, 3])

    @staticmethod
    def get_resnet101():
        return ResNet([3, 4, 23, 3])

    @staticmethod
    def get_resnet152():
        return ResNet([3, 8, 36, 3])


if __name__ == '__main__':
    test_data = torch.zeros([1, 224, 224, 3])
    model = ResNet.get_resnet50()
    from torchsummary import summary

    summary(model, input_size=(3, 256, 256))
