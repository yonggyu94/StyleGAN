import os

import torch
from torchvision.utils import save_image, make_grid
import torch.optim.lr_scheduler as lr_scheduler

from model import Generator
from model import Discriminator
from model import MappingNetwork

from dataloader import data_loader
from utils import cycle, WGANLoss, StyleGANLoss
from torch.nn import DataParallel

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'


class Solver():
    def __init__(self, config, channel_list):
        # Config - Model
        self.z_dim = config.z_dim
        self.channel_list = channel_list

        # Config - Training
        self.batch_size = config.batch_size
        self.g_lr = config.g_lr
        self.d_lr = config.d_lr
        self.decay_ratio = config.decay_ratio
        self.decay_iter = config.decay_iter
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.n_critic = config.n_critic
        self.lambda_value = config.lambda_value
        self.max_iter = config.max_iter
        self.loss_name = config.loss_name

        # Config - Test
        self.fixed_z = torch.randn(512, config.z_dim).to(dev)

        # Config - Path
        self.data_root = config.data_root
        self.log_root = config.log_root
        self.model_root = config.model_root
        self.sample_root = config.sample_root

        # Config - Miscellanceous
        self.print_loss_iter = config.print_loss_iter
        self.save_image_iter = config.save_image_iter
        self.save_parameter_iter = config.save_parameter_iter
        self.save_log_iter = config.save_log_iter

        self.writer = SummaryWriter(self.log_root)

    def build_model(self):
        self.G = Generator(channel_list=self.channel_list)
        self.G_ema = Generator(channel_list=self.channel_list)
        self.D = Discriminator(channel_list=self.channel_list)
        self.M = MappingNetwork(z_dim=self.z_dim)

        self.G = DataParallel(self.G).to(dev)
        self.G_ema = DataParallel(self.G_ema).to(dev)
        self.D = DataParallel(self.D).to(dev)
        self.M = DataParallel(self.M).to(dev)

        G_M_params = list(self.G.parameters()) + list(self.M.parameters())

        self.g_optimizer = torch.optim.Adam(params=G_M_params, lr=self.g_lr, betas=[self.beta1, self.beta2])
        self.d_optimizer = torch.optim.Adam(params=self.D.parameters(), lr=self.d_lr, betas=[self.beta1, self.beta2])

        self.g_scheduler = lr_scheduler.StepLR(self.g_optimizer, step_size=self.decay_iter, gamma=self.decay_ratio)
        self.d_scheduler = lr_scheduler.StepLR(self.d_optimizer, step_size=self.decay_iter, gamma=self.decay_ratio)

        print("Print model G, D")
        print(self.G)
        print(self.D)

    def build_loss(self):
        if self.loss_name == "stylegan":
            self.loss = StyleGANLoss(self.D)
        elif self.loss_name == "wgan-gp":
            self.loss = WGANLoss(self.D)

    def load_model(self, pkl_path, channel_list):
        ckpt = torch.load(pkl_path)

        self.G = Generator(channel_list=channel_list)
        self.G_ema = Generator(channel_list=channel_list)
        self.D = Discriminator(channel_list=channel_list)
        self.M = MappingNetwork(z_dim=self.z_dim)

        self.G = DataParallel(self.G).to(dev)
        self.G_ema = DataParallel(self.G_ema).to(dev)
        self.D = DataParallel(self.D).to(dev)
        self.M = DataParallel(self.M).to(dev)

        self.G.load_state_dict(ckpt["G"])
        self.G_ema.load_state_dict(ckpt["G_ema"])
        self.D.load_state_dict(ckpt["D"])
        self.M.load_state_dict(ckpt["M"])

    def save_model(self, iters, step):
        file_name = 'ckpt_%d_%d.pkl' % ((2*(2**(step+1)), iters))
        ckpt_path = os.path.join(self.model_root, file_name)
        ckpt = {
            'M': self.M.state_dict(),
            'G': self.G.state_dict(),
            'G_ema': self.G_ema.state_dict(),
            'D': self.D.state_dict()
        }
        torch.save(ckpt, ckpt_path)

    def save_img(self, iters, fixed_w, step):
        img_path = os.path.join(self.sample_root, "%d_%d.png" % (2*(2**(step+1)), iters))
        with torch.no_grad():
            fixed_w = fixed_w[:self.batch_size*2]
            w1, w2 = torch.split(fixed_w, self.batch_size, dim=0)
            const = torch.ones(self.batch_size, 512, 4, 4).to(dev)
            generated_imgs = self.G_ema(const, w1, w2, step, 1)
            save_image(make_grid(generated_imgs.cpu()/2+1/2, nrow=4, padding=2), img_path)

    def reset_grad(self):
        self.g_optimizer.zero_grad()
        self.d_optimizer.zero_grad()

    def lr_update(self):
        self.g_scheduler.step()
        self.d_scheduler.step()

    def set_phase(self, mode="train"):
        if mode == "train":
            self.G.train()
            self.G_ema.train()
            self.D.train()
            self.M.train()

        elif mode == "test":
            self.G.eval()
            self.G_ema.eval()
            self.D.eval()
            self.M.eval()

    def exponential_moving_average(self, beta=0.999):
        with torch.no_grad():
            G_param_dict = dict(self.G.named_parameters())
            for name, g_ema_param in self.G_ema.named_parameters():
                g_param = G_param_dict[name]
                g_ema_param.copy_(beta * g_ema_param + (1. - beta) * g_param)

    def train(self):
        # build model
        self.build_model()
        self.build_loss()

        for step in range(len(self.channel_list)):
            self.batch_size = self.batch_size // 2
            loader = data_loader(self.data_root, self.batch_size, img_size=2 * (2 ** (step + 1)))
            loader = iter(cycle(loader))

            if step == 0 or step == 1 or step == 2:
                self.max_iter = 20000
            elif step == 3 or step == 4 or step == 5:
                self.max_iter = 50000
            else:
                self.max_iter = 100000

            alpha = 0.0

            for iters in tqdm(range(self.max_iter+1)):
                real_img = next(loader)
                real_img = real_img.to(dev)
                # ===============================================================#
                #                    1. Train the discriminator                  #
                # ===============================================================#
                self.set_phase(mode="train")
                self.reset_grad()

                # Generate fake images
                const = torch.ones(self.batch_size, 512, 4, 4).to(dev)
                z = torch.randn(2 * self.batch_size, self.z_dim).to(dev)
                w = self.M(z)
                w1, w2 = torch.split(w, self.batch_size, dim=0)
                fake_img = self.G(const, w1, w2, step, alpha)

                # Compute d_loss
                d_real_loss, d_fake_loss, regularization = self.loss.d_loss(real_img, fake_img.detach(), step, alpha,
                                                                            self.lambda_value)

                # Backward and optimize.
                d_loss = d_real_loss + d_fake_loss + regularization
                d_loss.backward()
                self.d_optimizer.step()

                # ===============================================================#
                #                      2. Train the Generator                    #
                # ===============================================================#

                if (iters + 1) % self.n_critic == 0:
                    self.reset_grad()

                    # Compute loss with fake images.
                    const = torch.ones(self.batch_size, 512, 4, 4).to(dev)
                    z = torch.randn(2 * self.batch_size, self.z_dim).to(dev)
                    w = self.M(z)
                    w1, w2 = torch.split(w, self.batch_size, dim=0)
                    fake_img = self.G(const, w1, w2, step, alpha)

                    g_loss = self.loss.g_loss(fake_img, step, alpha)

                    # Backward and optimize.
                    g_loss.backward()
                    self.g_optimizer.step()

                # ===============================================================#
                #                   3. Save parameters and images                #
                # ===============================================================#
                # self.lr_update()
                torch.cuda.synchronize()
                alpha += 1 / (self.max_iter // 2)
                self.set_phase(mode="test")
                self.exponential_moving_average()

                # Print total loss
                if iters % self.print_loss_iter == 0:
                    print("Step : [%d/%d], Iter : [%d/%d], D_loss : [%.3f, %.3f, %.3f., %.3f], G_loss : %.3f" % (
                        step, len(self.channel_list)-1, iters, self.max_iter, d_loss.item(), d_real_loss.item(),
                        d_fake_loss.item(), regularization.item(), g_loss.item()
                    ))

                # Save generated images.
                if iters % self.save_image_iter == 0:
                    fixed_w = self.M(self.fixed_z)
                    self.save_img(iters, fixed_w, step)

                # Save the G and D parameters.
                if iters % self.save_parameter_iter == 0:
                    self.save_model(iters, step)

                # Save the logs on the tensorboard.
                if iters % self.save_log_iter == 0:
                    self.writer.add_scalar('g_loss/g_loss', g_loss.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_total', d_loss.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_real', d_real_loss.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_fake', d_fake_loss.item(), iters)
                    self.writer.add_scalar('d_loss/d_loss_gp', regularization.item(), iters)
