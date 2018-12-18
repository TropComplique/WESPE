import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):

    def __init__(self, image_size, num_input_channels):
        super(Discriminator, self).__init__()

        # final spatial image size
        height, width = image_size
        assert height % 16 == 0 and width % 16 == 0
        self.final_area = (height // 16) * (width // 16)

        feature_extractor = [
            nn.Conv2d(num_input_channels, 24, 11, stride=4, padding=5, bias=False),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(24, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        ]
        classifier = [
            nn.Linear(32 * self.final_area, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(128, 1)
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
        b = x.size(0)
        x = 2.0*x - 1.0
        x = self.feature_extractor(x)
        x = x.view(b, 32 * self.final_area)
        x = self.classifier(x).view(b)
        return x


class DiscriminatorSN(nn.Module):

    def __init__(self, image_size, num_input_channels):
        super(DiscriminatorSN, self).__init__()

        # final spatial image size
        height, width = image_size
        assert height % 16 == 0 and width % 16 == 0
        final_size = (height // 16, width // 16)

        feature_extractor = [
            spectral_norm(nn.Conv2d(num_input_channels, 24, 11, stride=4, padding=5)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(24, 32, 5, stride=2, padding=2)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(32, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 64, 3, stride=1, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(64, 32, 3, stride=2, padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AvgPool2d(final_size),
        ]
        classifier = [
            spectral_norm(nn.Linear(32, 32)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Linear(32, 1))
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
        b = x.size(0)
        x = 2.0*x - 1.0
        x = self.feature_extractor(x)
        x = x.squeeze(2).squeeze(2)  # shape [b, 32]
        x = self.classifier(x).view(b)
        return x