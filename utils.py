import os
import torch
import torch.nn.functional as F


dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def make_folder(exp_dir, log_dir, model_dir, sample_dir):
    log_root = os.path.join(exp_dir, log_dir)
    model_root = os.path.join(exp_dir, model_dir)
    sample_root = os.path.join(exp_dir, sample_dir)

    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
    if not os.path.isdir(log_root):
        os.makedirs(log_root, exist_ok=True)
    if not os.path.isdir(model_root):
        os.makedirs(model_root, exist_ok=True)
    if not os.path.isdir(sample_root):
        os.makedirs(sample_root, exist_ok=True)

    return log_root, model_root, sample_root


class WGANLoss(object):
    def __init__(self, dis):
        self.D = dis

    def d_loss(self, real_img, fake_img, step, alpha, gp_lambda=10.0):
        d_real_out = self.D(real_img, step, alpha)
        d_real_loss = - d_real_out.mean()

        d_fake_out = self.D(fake_img, step, alpha)
        d_fake_loss = d_fake_out.mean()

        d_gp_loss = self._penalty(real_img, fake_img, step, alpha)

        return d_real_loss, d_fake_loss, d_gp_loss * gp_lambda

    def g_loss(self, fake_img, step, alpha):
        d_fake_out = self.D(fake_img, step, alpha)
        g_loss = - d_fake_out.mean()
        return g_loss

    def _penalty(self, real_img, fake_img, step, alpha):
        beta = torch.rand(real_img.size(0), 1, 1, 1).to(dev)
        x_hat = (beta * real_img.data + (1 - beta) * fake_img.data).requires_grad_(True)
        d_x_hat_out = self.D(x_hat, step, alpha)

        weight = torch.ones(d_x_hat_out.size()).to(dev)
        dydx = torch.autograd.grad(outputs=d_x_hat_out,
                                   inputs=x_hat,
                                   grad_outputs=weight,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)


class StyleGANLoss(object):
    def __init__(self, dis):
        self.D = dis

    def d_loss(self, real_img, fake_img, step, alpha, r1_lambda=5.0):
        d_real_out = self.D(real_img, step, alpha)
        d_real_loss = F.softplus(-d_real_out).mean()

        d_fake_out = self.D(fake_img, step, alpha)
        d_fake_loss = F.softplus(d_fake_out).mean()

        r1_loss = self._r1_regularization(real_img, step, alpha)
        return d_real_loss, d_fake_loss, r1_loss * r1_lambda

    def g_loss(self, fake_img, step, alpha):
        d_fake_out = self.D(fake_img, step, alpha)
        g_loss = F.softplus(-d_fake_out).mean()
        return g_loss

    def _r1_regularization(self, real_img, step, alpha):
        real_img.requires_grad = True
        d_real_out = self.D(real_img, step, alpha)
        grad_real = torch.autograd.grad(outputs=d_real_out.sum(), inputs=real_img,
                                        create_graph=True)[0]
        r1_loss = grad_real.pow(2).view(grad_real.size(0), -1).sum(1).mean()
        return r1_loss
