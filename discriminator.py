import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):

    def __init__(self, image_size, num_input_channels, use_spectral_normalization=False):
        super(Discriminator, self).__init__()

        # final spatial image size
        self.final_area = (image_size // 16) ** 2

        feature_extractor = [
            nn.Conv2d(num_input_channels, 48, 11, stride=4, padding=5),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(48, 128, 5, stride=2, padding=2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True),
            nn.Conv2d(128, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(192, 192, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(192, affine=True),
            nn.Conv2d(128, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(128, affine=True)
        ]
        classifier = [
            nn.Linear(128 * self.final_area, 1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(1024, 1)
        ]

        # see https://arxiv.org/abs/1802.05957 for the details.
        if use_spectral_normalization:
            feature_extractor = [
                spectral_norm(m) if isinstance(m, nn.Conv2d) else m
                for m in feature_extractor
            ]
            classifier = [
                spectral_norm(m) if isinstance(m, nn.Linear) else m
                for m in classifier
            ]

        self.feature_extractor = nn.ModuleList(feature_extractor)
        self.classifier = nn.ModuleList(classifier)

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, num_input_channels, h, w].
            It has values in [0, 1] range.
        Returns:
            a float tensor with shape [b].
        """
        b = x.size(0)
        x = self.feature_extractor(x)
        x = x.view(b, 128 * self.final_area)
        x = self.classifier(x).view(b)
        return x
