import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from input_pipeline import PairDataset
from model import Generator


def main():

    dataset = PairDataset(
        first_dir='',
        second_dir='',
        num_samples=10000
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=32, shuffle=True,
        num_workers=1, pin_memory=True
    )
    generator = Generator().cuda()
    optimizer = optim.Adam(lr=5e-4, params=generator.parameters())

    for epoch in range(10):
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

            print('e:{0} i:{1} loss: {2:.3f}'.format(epoch, i, loss.item()))

    torch.save(generator.state_dict(), 'models/pretrained_generator.pth')


main()
