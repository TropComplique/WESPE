import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torchvision.models import vgg19
from scipy import signal


def gradient_penalty(x, y, f):
    """
    Arguments:
        x, y: float tensors with shape [b, c, h, w].
        f: a pytorch module.
    Returns:
        a float tensor with shape [].
    """

    # interpolation
    b = x.size(0)
    alpha = torch.rand([b, 1, 1, 1]).to(x.device)
    z = x + alpha * (y - x)
    z.requires_grad = True

    # compute gradient
    ones = torch.ones_like(z)
    g = grad(f(z), z, grad_outputs=ones, create_graph=True, only_inputs=True)[0]
    # it has shape [b, c, h, w]

    g = g.view(b, -1)
    return ((g.norm(p=2, dim=1) - 1.0)**2).mean(0)


def get_kernel(size=21, std=3):
    """Returns a 2D Gaussian kernel array."""
    k = signal.gaussian(size, std=std).reshape(size, 1)
    k = np.outer(k, k)
    return k/k.sum()


class GaussianBlur(nn.Module):

    def __init__(self):
        super(GaussianBlur, self).__init__()
        kernel = get_kernel(size=11, std=3)
        kernel = torch.Tensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0].unsqueeze(1)
        x2 = x[:, 1].unsqueeze(1)
        x3 = x[:, 2].unsqueeze(1)
        x1 = F.conv2d(x1, self.weight, padding=5)
        x2 = F.conv2d(x2, self.weight, padding=5)
        x3 = F.conv2d(x3, self.weight, padding=5)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


class GrayLayer(nn.Module):

    def __init__(self):
        super(GrayLayer, self).__init__()

    def forward(self, x):
        result = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
        return result.unsqueeze(1)


class ContentLoss(nn.Module):

    def __init__(self):
        super(ContentLoss, self).__init__()
        self.model = vgg19(pretrained=True).features[:-1]
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda().view(1, 3, 1, 1)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        """
        Arguments:
            x, y: float tensors with shape [b, 3, h, w].
            They represent RGB images with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """
        b = x.size(0)
        x = torch.cat([x, y], dim=0)
        x = (x - self.mean)/self.std

        x = self.model(x)
        # relu_5_4 features,
        # a float tensor with shape [2 * b, 512, h/16, w/16]
        
        x, y = torch.split(x, b, dim=0)
        b, c, h, w = x.size()
        normalizer = b * c * h * w
        return ((x - y)**2).sum()/normalizer


class TVLoss(nn.Module):

    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        """
        Arguments:
            x: a float tensor with shape [b, 3, h, w].
            It represents a RGB image with pixel values in [0, 1] range.
        Returns:
            a float tensor with shape [].
        """

        b, c, h, w = x.size()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :(h - 1), :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :(w - 1)]), 2).sum()
        return (h_tv + w_tv)/(b * c * h * w)
