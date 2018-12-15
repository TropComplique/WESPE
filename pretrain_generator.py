import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from input_pipeline import PairDataset
from model import Generator


NUM_STEPS = 5000
IMAGE_SIZE = 64
BATCH_SIZE = 32


def main():

    dataset = PairDataset(
        first_dir='',
        second_dir='',
        num_samples=NUM_STEPS * BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True
    )
    generator = Generator().cuda()
    optimizer = optim.Adam(lr=5e-4, params=generator.parameters())

    for i, (x, y) in enumerate(data_loader):

        if np.random.rand() > 0.5:
            x = x.cuda()
        else:
            x = y.cuda()

        restored_x = generator(x)
        batch_size = x.size(0)
        loss = ((restored_x - x)**2).sum()/batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('i:{0} loss: {1:.3f}'.format(i, loss.item()))

    torch.save(generator.state_dict(), 'models/pretrained_generator.pth')


main()
