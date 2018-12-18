import torch
import torch.nn as nn
import torch.optim as optim

from generators import GeneratorSN
from discriminators import DiscriminatorSN
from utils import gradient_penalty, ContentLoss, TVLoss


GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 4e-4


class GAN:

    def __init__(self, image_size):

        self.generator_g = GeneratorSN().cuda()
        self.generator_f = GeneratorSN().cuda()
        self.discriminator = DiscriminatorSN(image_size, num_input_channels=3).cuda()

        self.content_criterion = ContentLoss().cuda()
        self.tv_criterion = TVLoss().cuda()
        self.realism_criterion = nn.BCEWithLogitsLoss().cuda()

        betas = (0.0, 0.9)
        self.g_optimizer = optim.Adam(self.generator_g.parameters(), lr=GENERATOR_LR, betas=betas)
        self.f_optimizer = optim.Adam(self.generator_f.parameters(), lr=GENERATOR_LR, betas=betas)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=DISCRIMINATOR_LR, betas=betas)

    def train_step(self, x, y, update_generator=True):

        y_fake = self.generator_g(x)
        x_fake = self.generator_f(y_fake)

        for p in self.discriminator.parameters():
            p.requires_grad = False

        content_loss = self.content_criterion(x, x_fake)
        tv_loss = self.tv_criterion(y_fake)

        batch_size = x.size(0)
        pos_labels = torch.ones(batch_size, dtype=torch.float, device=x.device)
        neg_labels = torch.zeros(batch_size, dtype=torch.float, device=x.device)
        realism_generation_loss = self.realism_criterion(self.discriminator(y_fake), pos_labels)

        if update_generator:

            generator_loss = content_loss + 100.0 * tv_loss
            generator_loss += 5e-2 * realism_generation_loss

            self.g_optimizer.zero_grad()
            self.f_optimizer.zero_grad()
            generator_loss.backward()
            self.g_optimizer.step()
            self.f_optimizer.step()

        for p in self.discriminator.parameters():
            p.requires_grad = True

        targets = torch.cat([pos_labels, neg_labels], dim=0)
        is_real_real = self.discriminator(y)
        is_fake_real = self.discriminator(y_fake.detach())
        logits = torch.cat([is_real_real, is_fake_real], dim=0)
        discriminator_loss = self.realism_criterion(logits, targets)

        self.d_optimizer.zero_grad()
        discriminator_loss.backward()
        self.d_optimizer.step()

        loss_dict = {
            'content': content_loss.item(),
            'tv': tv_loss.item(),
            'realism_generation': realism_generation_loss.item(),
            'discriminator': discriminator_loss.item(),
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.generator_f.state_dict(), model_path + '_generator_f.pth')
        torch.save(self.generator_g.state_dict(), model_path + '_generator_g.pth')
        torch.save(self.discriminator.state_dict(), model_path + '_discriminator_t.pth')


"""
for wasserstein gan training:
self.color_criterion = lambda x, y: (-(y*x) + (1.0 - y)*x).mean(0)
self.texture_criterion = lambda x, y: (-(y*x) + (1.0 - y)*x).mean(0)
lambda_constant = 1.0
gp1 = gradient_penalty(y_real_blur, y_fake_blur.detach(), self.discriminator_c)
gp2 = gradient_penalty(y_real_gray, y_fake_gray.detach(), self.discriminator_t)
discriminator_loss += lambda_constant * (gp1 + gp2)
"""
