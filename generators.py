import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ResBlock(nn.Module):

    def __init__(self):
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
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
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 9, padding=4)
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        """
        x = 2.0*x - 1.0
        x = self.beginning(x)
        x = self.blocks(x)
        x = self.additional(x)
        x = 0.5 * torch.tanh(x) + 0.5
        return x


class ResBlockSN(nn.Module):

    def __init__(self):
        super(ResBlockSN, self).__init__()

        self.layers = nn.Sequential(
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 3, padding=1))
        )

    def forward(self, x):
        return x + self.layers(x)


class GeneratorSN(nn.Module):

    def __init__(self):
        super(GeneratorSN, self).__init__()
        self.beginning = spectral_norm(nn.Conv2d(3, 64, 9, padding=4))
        self.blocks = nn.Sequential(
            ResBlockSN(),
            ResBlockSN(),
            ResBlockSN(),
            ResBlockSN(),
        )
        self.additional = nn.Sequential(
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 3, padding=1)),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(64, 3, 9, padding=4))
        )

        def weights_init(m):
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(weights_init)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        """
        x = 2.0*x - 1.0
        x = self.beginning(x)
        x = self.blocks(x)
        x = self.additional(x)
        x = 0.5 * torch.tanh(x) + 0.5
        return x
