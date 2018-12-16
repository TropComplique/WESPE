import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):

    def __init__(self, image_size, num_input_channels):
        super(Discriminator, self).__init__()

        # final spatial image size
        assert image_size % 16 == 0
        self.final_size = (image_size // 16) ** 2

        feature_extractor = [
            nn.Conv2d(num_input_channels, 24, 11, stride=4, padding=5, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(24, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 96, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(96, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        classifier = [
            nn.Linear(64 * self.final_size, 256, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 1)
        ]

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, num_input_channels, h, w].
            It has values in [0, 1] range.
        Returns:
            a float tensor with shape [b].
        """
        x = 2.0*x - 1.0
        x = self.feature_extractor(x)
        x = x.view(x.size(0), 64 * self.final_size)
        x = self.classifier(x).view(b)
        return x


class DiscriminatorSN(nn.Module):

    def __init__(self, image_size, num_input_channels):

        # final spatial image size
        assert image_size % 16 == 0
        self.final_size = (image_size // 16) ** 2

        feature_extractor = [
            spectral_norm(nn.Conv2d(num_input_channels, 24, 11, stride=4, padding=5)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(24, 64, 5, stride=2, padding=2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 96, 3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(96, 96, 3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(96, 64, 3, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        classifier = [
            spectral_norm(nn.Linear(64 * self.final_size, 256)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Linear(256, 1))
        ]

        self.feature_extractor = nn.Sequential(*feature_extractor)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, num_input_channels, h, w].
            It has values in [0, 1] range.
        Returns:
            a float tensor with shape [b].
        """
        x = 2.0*x - 1.0
        x = self.feature_extractor(x)
        x = x.view(x.size(0), 64 * self.final_size)
        x = self.classifier(x).view(b)
        return x
