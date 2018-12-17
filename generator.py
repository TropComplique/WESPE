import torch
import torch.nn as nn
import torch.nn.functional as F


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


class NoiseGenerator(nn.Module):

    def __init__(self, image_size):
        super(NoiseGenerator, self).__init__()
        self.beginning = nn.Conv2d(3, 64, 9, padding=4)

        def block():
            return nn.Sequential(
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 3, 3, padding=1, bias=False)
            )

        self.blocks = nn.ModuleList(9*[block()])

        final_size = image_size // 8
        self.weights = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.AvgPool2d(final_size)
            nn.Conv2d(32, 10, 1)
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
        weights = self.weights(x)  # shape [b, 10, 1, 1]
        weights = F.softmax(weights, dim=1)
        weights = weights.unsqueeze(4)  # shape [b, 10, 1, 1, 1]

        x_initial = x
        b, c, h, w = x.size()
        z = torch.randn(b, h, w)

        x = 2.0*x - 1.0
        x = self.beginning(x)

        result = [x_initial]
        for b in self.blocks:
            result.append(0.5 * torch.tanh(b(x)) + 0.5)

        result = torch.stack(result, dim=1)  # shape [b, 10, 3, h, w]
        return (weights * result).sum(1)
