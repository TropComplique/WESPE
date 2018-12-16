import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from torch.utils.data import DataLoader
from input_pipeline import PairDataset
from wespe import WESPE
import json


NUM_STEPS = 15000
IMAGE_SIZE = 96
BATCH_SIZE = 32
MODEL_SAVE_PREFIX = 'models/run00'
TRAIN_LOGS = 'losses_run00.json'


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
    wespe = WESPE(IMAGE_SIZE)

    logs = []
    text = 'i: {0}, content: {1:.3f}, tv: {2:.5f}, ' +\
           'color generation: {3:.3f}, texture generation: {4:.3f}, ' +\
           'color discriminator: {5:.3f}, texture discriminator: {6:.3f}'

    for i, (x, y) in enumerate(data_loader):

        x = x.cuda()
        y = y.cuda()
        losses = wespe.train_step(x, y)

        log = text.format(
            i, losses['content'], losses['tv'],
            losses['color_generation'], losses['texture_generation'],
            losses['color_discriminator'], losses['texture_discriminator']
        )
        print(log)
        logs.append(losses)

        if (i + 1) % 1000 == 0:
            wespe.save_model(MODEL_SAVE_PREFIX)
            with open(TRAIN_LOGS, 'w') as f:
                json.dump(logs, f)


main()
