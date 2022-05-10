from torch import nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same")
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same")

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same")
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding="same")

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding="same")
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same")
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding="same")

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding="same")
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same")
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same")

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same")
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same")
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding="same")

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc1 = nn.Linear((input_size // 32) ** 2 * 512, 4096)
        # self.fc2 = nn.Linear(4096, 4096)
        # self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        # 2s
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        # 4s
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        # 8s
        x = self.maxpool(x)
        pool_8 = x

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        # 16s
        x = self.maxpool(x)
        pool_16 = x

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        # 32s
        x = self.maxpool(x)
        pool_32 = x

        return {"pool32": pool_32,
                "pool16": pool_16,
                "pool8": pool_8}


def test():
    vgg = VGG16()
    from torchsummary import summary
    summary(vgg, input_size=(32, 256, 256))


if __name__ == '__main__':
    test()
