import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):

    def __init__(self):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.instance_norm1 = nn.InstanceNorm2d(64, affine=True)
        self.instance_norm2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        y = F.relu(self.instance_norm1(self.conv1(x)))
        y = F.relu(self.instance_norm2(self.conv2(y))) + x
        return y


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 9, padding=4)
        self.blocks = nn.Sequential(
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
            ConvBlock(),
        )
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        """
        x_initial = x
        x = F.relu(self.conv1(x))
        x = self.blocks(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = 0.5 * torch.tanh(self.conv4(x)) + 0.5
        x = 0.5 * (x_initial + x)
        return x
