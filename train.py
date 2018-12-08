from torch.utils.data import DataLoader
from input_pipeline import PairDataset
from model import WESPE


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
    wespe = WESPE()

    for epoch in range(50):
        for i, (x, y) in enumerate(data_loader):

            x = x.cuda()
            y = y.cuda()
            losses = wespe.train_step(x, y)

            text = 'e:{0}, i:{1}, content loss: {2:.3f}, tv loss: {3:.3f}, gen_color_loss: {4:.3f}, ' +\
                   'gen_texture_loss: {5:.3f}, discri_color_loss: {6:.3f}, discri_texture_loss: {7:.3f}'
            print(text.format(
                epoch, i, losses['content'], losses['tv'], losses['gen_dc'],
                losses['gen_dt'], losses['color_loss'], losses['texture_loss']
            ))
        wespe.save_model('models/run01')


main()
