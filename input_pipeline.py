import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


IMAGE_SIZE = 112


transform = transforms.Compose([
    transforms.RandomCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
])


class PairDataset(Dataset):

    def __init__(self, first_dir, second_dir, num_samples):
        """
        Arguments:
            first_dir, second_dir: strings, paths to folders with images.
            num_samples: an integer.
        """

        self.names1 = os.listdir(first_dir)
        self.names2 = os.listdir(second_dir)
        self.first_dir = first_dir
        self.second_dir = second_dir
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        """
        Get a random pair of image crops.
        It returns a tuple of float tensors with shape [3, IMAGE_SIZE, IMAGE_SIZE].
        They represent RGB images with pixel values in [0, 1] range.
        """
        i = np.random.randint(0, len(self.names1))
        j = np.random.randint(0, len(self.names2))
        name1, name2 = self.names1[i], self.names2[j]
        image1 = Image.open(os.path.join(self.first_dir, name1))
        image2 = Image.open(os.path.join(self.second_dir, name2))
        return transform(image1), transform(image2)
