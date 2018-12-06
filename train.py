import numpy as np
import torch.optim as optimizer
from torch.utils.data import DataLoader
from model import WESPE
from input_pipeline import PairDataset


def main():

    wespe = WESPE()

    dataset = PairDataset(
        first_dir='',
        second_dir='',
        num_samples=1000
    )

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=30, shuffle=True,
        num_workers=1, pin_memory=True
    )

    for epoch in range(10):
        
        for i, (x, y) in enumerate(data_loader):

            x = x.cuda()
            y = y.cuda()

            loss = wespe.train_step(x, y)

            print("e:{} i:{} content loss: {}, tv loss:{}, gen_color_loss: {}, gen_texture_loss:{}, "
                  "discri_color_loss: {}, discri_texture_loss: {}".format(epoch, i, loss['content'],
                                                                          loss['tv'], loss['gen_dc'],
                                                                          loss['gen_dt'], loss['color_loss'],
                                                                          loss['texture_loss']))
        wespe.save_model('models/')


main()
