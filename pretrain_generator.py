import numpy as np
import torch.optim as optimizer
from torch.utils.data import DataLoader
from model import Generator
from data_provider import PairDataset


def main():

    generator = Generator().cuda()

    dataset = PairDataset(
        first_dir='',
        second_dir=''
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.worker,
        pin_memory=True,
    )

    for epoch in range(args.epoches):
        train_iter = iter(data_loader)
        for i in range(len(data_loader)):

            x, y = next(train_iter)
            if args.use_cuda:
                x = x.cuda()
                y = y.cuda()

            loss = wespe.train_step(x, y)

            print("e:{} i:{} content loss: {}, tv loss:{}, gen_color_loss: {}, gen_texture_loss:{}, "
                  "discri_color_loss: {}, discri_texture_loss: {}".format(epoch, i, loss['content'],
                                                                          loss['tv'], loss['gen_dc'],
                                                                          loss['gen_dt'], loss['color_loss'],
                                                                          loss['texture_loss']))
        wespe.save_model(args.save_model_path)


if __name__ == '__main__':
    main()
