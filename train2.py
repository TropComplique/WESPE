import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import DataLoader
from input_pipeline import PairDataset
from wespe2 import WESPE
import json


NUM_STEPS = 50000
IMAGE_SIZE = 96
BATCH_SIZE = 64
MODEL_SAVE_PREFIX = 'models/run02'
TRAIN_LOGS = 'losses_run02.json'
N_DISCRIMINATOR = 1


def main():

    dataset = PairDataset(
        
        num_samples=NUM_STEPS * BATCH_SIZE,
        image_size=IMAGE_SIZE
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=1, pin_memory=True
    )
    wespe = WESPE(IMAGE_SIZE)

    logs = []
    text = 'i: {0}, content: {1:.3f}, tv: {2:.5f}, ' +\
           'realism: {3:.3f}, discriminator: {4:.3f}'

    for i, (x, y) in enumerate(data_loader):

        x = x.cuda()
        y = y.cuda()

        update_generator = i % N_DISCRIMINATOR == 0
        losses = wespe.train_step(x, y, update_generator)

        log = text.format(
            i, losses['content'], losses['tv'],
            losses['realism_generation'], losses['discriminator']
        )
        print(log)
        logs.append(losses)

        if (i + 1) % 1000 == 0:
            wespe.save_model(MODEL_SAVE_PREFIX)
            with open(TRAIN_LOGS, 'w') as f:
                json.dump(logs, f)


main()
