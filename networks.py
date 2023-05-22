from typing import List

from torch import nn

class APNN(nn.Module):

    def __init__(self, input_channels : int, kernels : List[int]):
        super().__init__()

        self.paddings = [int((kernel - 1)/2) for kernel in kernels]

        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=48,
                               kernel_size=kernels[0],
                               stride=1,
                               padding=self.paddings[0])
        self.conv2 = nn.Conv2d(in_channels=48,
                               out_channels=32,
                               kernel_size=kernels[1],
                               stride=1,
                               padding=self.paddings[1])
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=input_channels-1,
                               kernel_size=kernels[2],
                               stride=1,
                               padding=self.paddings[2])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out + x[:, :-1, :, :]

class DRPNN(nn.Module):

    def __init__(self, input_channels : int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv2 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv5 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv6 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv7 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv8 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv9 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=7,
                               stride=1,
                               padding=(3, 3))
        self.conv10 = nn.Conv2d(in_channels=64,
                                out_channels=input_channels,
                                kernel_size=7,
                                stride=1,
                                padding=(3, 3))
        self.conv11 = nn.Conv2d(in_channels=input_channels,
                                out_channels=input_channels-1,
                                kernel_size=3,
                                stride=1,
                                padding=(1, 1))
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        out = self.relu(self.conv5(out))
        out = self.relu(self.conv6(out))
        out = self.relu(self.conv7(out))
        out = self.relu(self.conv8(out))
        out = self.relu(self.conv9(out))
        out = self.conv10(out)
        out = self.conv11(self.relu(out + x))
        return out