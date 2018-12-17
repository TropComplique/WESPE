import torch
import torch.nn as nn
import torch.optim as optim

from generator import Generator
from discriminator import Discriminator, DiscriminatorSN
from utils import GaussianBlur, Grayscale, Sobel
from utils import gradient_penalty, ContentLoss, TVLoss


GENERATOR_LR = 1e-4
DISCRIMINATOR_LR = 4e-4


class WESPE:

    def __init__(self, image_size):

        self.generator_g = Generator().cuda()
        self.generator_f = Generator().cuda()
        self.discriminator_c = DiscriminatorSN(image_size, num_input_channels=3).cuda()
        self.discriminator_t = DiscriminatorSN(image_size, num_input_channels=1).cuda()

        self.content_criterion = ContentLoss().cuda()
        self.tv_criterion = TVLoss().cuda()
        self.color_criterion = nn.BCEWithLogitsLoss().cuda()
        self.texture_criterion = nn.BCEWithLogitsLoss().cuda()

        # for wasserstein gan training
        # self.color_criterion = lambda x, y: (-(y*x) + (1.0 - y)*x).mean(0)
        # self.texture_criterion = lambda x, y: (-(y*x) + (1.0 - y)*x).mean(0)

        self.g_optimizer = optim.Adam(lr=GENERATOR_LR, params=self.generator_g.parameters(), betas=(0.0, 0.9))
        self.f_optimizer = optim.Adam(lr=GENERATOR_LR, params=self.generator_f.parameters(), betas=(0.0, 0.9))
        self.c_optimizer = optim.Adam(lr=DISCRIMINATOR_LR, params=self.discriminator_c.parameters(), betas=(0.0, 0.9))
        self.t_optimizer = optim.Adam(lr=DISCRIMINATOR_LR, params=self.discriminator_t.parameters(), betas=(0.0, 0.9))

        self.blur = GaussianBlur().cuda()
        self.gray = Grayscale().cuda()

    def train_step(self, x, y, update_generator=True):

        y_fake = self.generator_g(x)
        x_fake = self.generator_f(y_fake)

        for p in self.discriminator_c.parameters():
            p.requires_grad = False
        for p in self.discriminator_t.parameters():
            p.requires_grad = False

        content_loss = self.content_criterion(x, x_fake)
        tv_loss = self.tv_criterion(y_fake)

        batch_size = x.size(0)
        pos_labels = torch.ones(batch_size, dtype=torch.float, device=x.device)
        neg_labels = torch.zeros(batch_size, dtype=torch.float, device=x.device)

        y_fake_blur = self.blur(y_fake)
        color_generation_loss = self.color_criterion(self.discriminator_c(y_fake_blur), pos_labels)

        y_fake_gray = self.gray(y_fake)
        texture_generation_loss = self.texture_criterion(self.discriminator_t(y_fake_gray), pos_labels)

        if update_generator:

            generator_loss = content_loss + 100.0 * tv_loss
            generator_loss += 5e-2 * (color_generation_loss + texture_generation_loss)

            self.g_optimizer.zero_grad()
            self.f_optimizer.zero_grad()
            generator_loss.backward()
            self.g_optimizer.step()
            self.f_optimizer.step()

        for p in self.discriminator_c.parameters():
            p.requires_grad = True
        for p in self.discriminator_t.parameters():
            p.requires_grad = True

        y_real_blur = self.blur(y)
        y_real_gray = self.gray(y)
        targets = torch.cat([pos_labels, neg_labels], dim=0)

        is_real_real = self.discriminator_c(y_real_blur)
        is_fake_real = self.discriminator_c(y_fake_blur.detach())
        logits = torch.cat([is_real_real, is_fake_real], dim=0)
        color_discriminator_loss = self.color_criterion(logits, targets)

        is_real_real = self.discriminator_t(y_real_gray)
        is_fake_real = self.discriminator_t(y_fake_gray.detach())
        logits = torch.cat([is_real_real, is_fake_real], dim=0)
        texture_discriminator_loss = self.texture_criterion(logits, targets)

        discriminator_loss = color_discriminator_loss + texture_discriminator_loss
        # lambda_constant = 1.0
        # gp1 = gradient_penalty(y_real_blur, y_fake_blur.detach(), self.discriminator_c)
        # gp2 = gradient_penalty(y_real_gray, y_fake_gray.detach(), self.discriminator_t)
        # discriminator_loss += lambda_constant * (gp1 + gp2)

        self.c_optimizer.zero_grad()
        self.t_optimizer.zero_grad()
        discriminator_loss.backward()
        self.c_optimizer.step()
        self.t_optimizer.step()

        loss_dict = {
            'content': content_loss.item(),
            'tv': tv_loss.item(),
            'color_generation': color_generation_loss.item(),
            'texture_generation': texture_generation_loss.item(),
            'color_discriminator': color_discriminator_loss.item(),
            'texture_discriminator': texture_discriminator_loss.item(),
        }
        return loss_dict

    def save_model(self, model_path):
        torch.save(self.generator_f.state_dict(), model_path + '_generator_f.pth')
        torch.save(self.generator_g.state_dict(), model_path + '_generator_g.pth')
        torch.save(self.discriminator_t.state_dict(), model_path + '_discriminator_t.pth')
        torch.save(self.discriminator_c.state_dict(), model_path + '_discriminator_c.pth')
