import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(64)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False)
        )

    def forward(self, x):
        return x + self.layers(x)


class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.beginning = nn.Conv2d(3, 64, 9, padding=4)
        self.blocks = nn.Sequential(
            ResBlock(),
            ResBlock(),
            ResBlock(),
            ResBlock(),
        )
        self.additional = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 9, padding=4)
        )

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
        x = 2.0*x - 1.0
        x = self.beginning(x)
        x = self.blocks(x)
        x = self.additional(x)
        x = 0.5 * torch.tanh(x) + 0.5
        x = 0.5 * (x_initial + x)
        return x
