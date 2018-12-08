import os
import random
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class PairDataset(Dataset):

    def __init__(self, first_dir, second_dir, num_samples, image_size):
        """
        Arguments:
            first_dir, second_dir: strings, paths to folders with images.
            num_samples: an integer.
            image_size: an integer.
        """

        self.names1 = os.listdir(first_dir)
        self.names2 = os.listdir(second_dir)
        self.first_dir = first_dir
        self.second_dir = second_dir
        self.num_samples = num_samples

        self.transform = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, _):
        """
        Get a random pair of image crops.
        It returns a tuple of float tensors with shape [3, image_size, image_size].
        They represent RGB images with pixel values in [0, 1] range.
        """
        i = np.random.randint(0, len(self.names1))
        j = np.random.randint(0, len(self.names2))
        name1, name2 = self.names1[i], self.names2[j]
        image1 = Image.open(os.path.join(self.first_dir, name1))
        image2 = Image.open(os.path.join(self.second_dir, name2))
        return self.transform(image1), self.transform(image2)
