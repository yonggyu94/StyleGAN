import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.utils.spectral_norm as spectral_norm
from torch.autograd import Function
import math

''' Device type'''
dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda'


class WrongNormException(Exception):
    def __str__(self):
        return 'You should choose \'BN\', \'IN\' or \'SN\''


class WrongNonLinearException(Exception):
    def __str__(self):
        return 'You should choose \'relu\', \'leaky_relu\' or \'tanh\''


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride, padding):
        super(EqualizedConv2d, self).__init__()
        conv = nn.Conv2d(in_ch, out_ch, k_size, stride, padding)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, x):
        out = self.conv(x)
        return out


class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EqualizedLinear, self).__init__()
        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, x):
        out = self.linear(x)
        return out


class AffineBlock(nn.Module):
    def __init__(self, in_dim, out_dim, n_slope=0.01, norm="SN", non_linear='relu'):
        super(AffineBlock, self).__init__()
        layers = []

        if norm == "SN":
            layers.append(spectral_norm(nn.Linear(in_dim, out_dim)))
        else:
            layers.append(nn.Linear(in_dim, out_dim))
            if norm == 'BN':
                layers.append(nn.BatchNorm1d(out_dim, affine=True))
            elif norm == 'IN':
                layers.append(nn.InstanceNorm1d(out_dim, affine=True))
            elif norm == None: pass
            else: raise WrongNormException()

        if non_linear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif non_linear == 'leaky_relu':
            layers.append(nn.LeakyReLU(n_slope, inplace=True))
        elif non_linear == 'sigmoid':
            layers.append(nn.Sigmoid(inplace=True))
        elif non_linear == 'tanh':
            layers.append(nn.Tanh(inplace=True))
        elif non_linear == None:
            pass
        else:
            raise WrongNonLinearException()

        self.affine_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.affine_block(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01,
                 norm='SN', non_linear='leaky_relu', equalized=True):
        super(ConvBlock, self).__init__()
        layers = []

        if norm == 'SN':
            layers.append(spectral_norm(nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding)))
        else:
            if equalized:
                layers.append(EqualizedConv2d(in_dim, out_dim, k_size=ksize, stride=stride, padding=padding))
            else:
                layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding))

            if norm == 'BN':
                layers.append(nn.BatchNorm2d(out_dim, affine=True))
            elif norm == 'IN':
                layers.append(nn.InstanceNorm2d(out_dim, affien=True))
            elif norm == 'PN':
                layers.append(PixelNorm())
            elif norm == None : pass
            else: raise WrongNormException

        if non_linear == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif non_linear == 'leaky_relu':
            layers.append(nn.LeakyReLU(n_slope, inplace=True))
        elif non_linear == 'sigmoid':
            layers.append(nn.Sigmoid(inplace=True))
        elif non_linear == 'tanh':
            layers.append(nn.Tanh(inplace=True))
        elif non_linear == None: pass
        else: raise WrongNonLinearException

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_dim, w_dim=512):
        super(AdaptiveInstanceNorm, self).__init__()
        self.in_dim = in_dim
        self.linear = nn.Linear(w_dim, self.in_dim*2)
        self.instance_norm = nn.InstanceNorm2d(self.in_dim*2)

    def forward(self, x, w):
        w_out = self.linear(w)
        w_out = w_out.unsqueeze(2).unsqueeze(3)
        out = self.instance_norm(x)         # [B, in_dim, H, W]

        style_mean, style_std = torch.split(w_out, self.in_dim, dim=1)      # [B, in_dim], [B, in_dim]
        out = out * style_mean.expand_as(out) + style_std.expand_as(out)    # [B, in_dim, H, W]
        return out


class NoiseInjection(nn.Module):
    def __init__(self, in_dim):
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1, in_dim, 1, 1))

    def forward(self, x):
        batch, c, h, w = x.shape
        noise = torch.randn(batch, c, h, w).to(dev)
        out = x + self.weight * noise
        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()

    def forward(self, x):
        out = x / torch.sqrt(torch.mean(x**2, dim=1, keepdim=True) + 1e-8)
        return out


class StdConcat(nn.Module):
    def __init__(self):
        super(StdConcat, self).__init__()

    def forward(self, x):
        mean_std = torch.mean(x.std(0))
        mean_std = mean_std.expand(x.size(0), 1, 4, 4)
        out = torch.cat([x, mean_std], dim=1)
        return out


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super(Blur, self).__init__()

        weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer('weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class Block(nn.Module):
    def __init__(self, in_dim, out_dim, ksize_1, pad_1, ksize_2, pad_2, norm="SN", n_slope=0.2,
                 non_linear='leaky_relu', equlized=False):
        super(Block, self).__init__()
        self.convblock_1 = ConvBlock(in_dim, in_dim, ksize=ksize_1, padding=pad_1, n_slope=n_slope,
                                     norm=norm, non_linear=non_linear, equalized=equlized)
        self.convblock_2 = ConvBlock(in_dim, out_dim, ksize=ksize_2, padding=pad_2,
                                     n_slope=n_slope, norm=norm, non_linear=non_linear,
                                     equalized=equlized)

    def forward(self, x):
        out = self.convblock_1(x)
        out = self.convblock_2(out)
        return out


class SynthesisConstBlock(nn.Module):
    def __init__(self, in_dim, out_dim, w_dim=512, norm="PN"):
        super(SynthesisConstBlock, self).__init__()
        self.noise_inject1 = NoiseInjection(in_dim=in_dim)
        self.adain1 = AdaptiveInstanceNorm(in_dim, w_dim)
        self.conv = ConvBlock(in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.2, norm=norm,
                              non_linear='leaky_relu', equalized=True)
        self.noise_inject2 = NoiseInjection(in_dim=out_dim)
        self.adain2 = AdaptiveInstanceNorm(out_dim, w_dim)

    def forward(self, const, w1, w2):
        out = self.noise_inject1(const)
        out = self.adain1(out, w1)
        out = self.conv(out)
        out = self.noise_inject2(out)
        out = self.adain2(out, w2)
        return out


class SynthesisBlock(nn.Module):
    def __init__(self, in_dim, out_dim, w_dim=512, norm="PN"):
        super(SynthesisBlock, self).__init__()
        self.conv1 = ConvBlock(in_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01,
                               norm=norm, non_linear='leaky_relu', equalized=True)
        self.blur = Blur(out_dim)

        self.noise_inject1 = NoiseInjection(in_dim=out_dim)
        self.adain1 = AdaptiveInstanceNorm(out_dim, w_dim)

        self.conv2 = ConvBlock(out_dim, out_dim, ksize=3, stride=1, padding=1, n_slope=0.01,
                               norm=norm, non_linear='leaky_relu', equalized=True)
        self.noise_inject2 = NoiseInjection(in_dim=out_dim)
        self.adain2 = AdaptiveInstanceNorm(out_dim, w_dim)

    def forward(self, x, w1, w2):
        out = self.conv1(x)
        out = self.blur(out)

        out = self.noise_inject1(out)
        out = self.adain1(out, w1)

        out = self.conv2(out)
        out = self.noise_inject2(out)
        out = self.adain2(out, w2)
        return out


class MappingNetwork(nn.Module):
    def __init__(self, z_dim, n_mapping=8, w_dim=512):
        super(MappingNetwork, self).__init__()
        self.z_dim = z_dim
        # self.fixed_z = torch.randn(batch_size, z_dim).to(dev)
        blocks = []
        for i in range(n_mapping):
            if i == 0:
                blocks.append(AffineBlock(z_dim, w_dim, norm="SN", non_linear='leaky_relu'))
            else:
                blocks.append(AffineBlock(w_dim, w_dim, norm="SN", non_linear='leaky_relu'))

        self.mapping_network = nn.Sequential(*blocks)

    def forward(self, z):
        out = self.mapping_network(z)
        return out


class Generator(nn.Module):
    def __init__(self, channel_list=[512, 512, 512, 512, 256, 128, 64, 32]):
        super(Generator, self).__init__()
        progress_layers = []
        to_rgb_layers = []
        for i in range(len(channel_list)):
            if i == 0:
                progress_layers.append(SynthesisConstBlock(channel_list[i], channel_list[i],
                                                           norm="PN"))
            else:
                progress_layers.append(SynthesisBlock(channel_list[i - 1], channel_list[i],
                                                      norm="PN"))
            to_rgb_layers.append(nn.Conv2d(channel_list[i], 3, 1))

        self.progress = nn.ModuleList(progress_layers)
        self.to_rgb = nn.ModuleList(to_rgb_layers)

    def forward(self, x, w1, w2, step=0, alpha=-1):
        for i, (block, to_rgb) in enumerate(zip(self.progress, self.to_rgb)):
            if i > 0:
                upsample = F.upsample(out, scale_factor=2)
                out = block(upsample, w1, w2)
            else:
                out = block(x, w1, w2)

            if i == step:
                out = to_rgb(out)
                if i != 0 and 0 <= alpha < 1:               # The first module does not need previous to_rgb module
                    skip_rgb = self.to_rgb[i-1](upsample)
                    out = (1-alpha)*skip_rgb + alpha*out
                break
        return out


class Discriminator(nn.Module):
    def __init__(self, channel_list=[512, 512, 512, 512, 256, 128, 64, 32], n_slope=0.2, n_label=10):
        super(Discriminator, self).__init__()
        reversed(channel_list)

        self.std_concat = StdConcat()

        progress_layers = []
        from_rgb_layers = []
        for i in range(len(channel_list) - 1, -1, -1):
            if i == 0:
                progress_layers.append(Block(channel_list[i] + 1, channel_list[i], 3, 1, 4, 0, norm="SN", equlized=False))
            else:
                progress_layers.append(Block(channel_list[i], channel_list[i - 1], 3, 1, 3, 1, norm="SN", equlized=False))
            from_rgb_layers.append(nn.Conv2d(3, channel_list[i], 1))

        self.progress = nn.ModuleList(progress_layers)
        self.from_rgb = nn.ModuleList(from_rgb_layers)

        self.n_layer = len(self.progress)

        linear = nn.Linear(512, 1)
        self.linear = equal_lr(linear)

    def forward(self, x, step=0, alpha=-1):
        step = self.n_layer - 1 - step
        for i in range(step, self.n_layer):
            if i == step:
                out = self.from_rgb[i](x)

            if i == (self.n_layer-1):
                out = self.std_concat(out)
                out = self.progress[i](out)
            else:
                out = self.progress[i](out)
                out = F.avg_pool2d(out, 2)

            if i == step:
                if i != 7 and 0 <= alpha < 1:
                    downsample = F.avg_pool2d(x, 2)
                    skip_rgb = self.from_rgb[i+1](downsample)
                    out = (1-alpha)*skip_rgb + alpha*out

        out = out.squeeze(3).squeeze(2)
        out = self.linear(out)

        return out[:, 0]


if __name__ == "__main__":
    z = torch.rand(8, 512).to(dev)
    img = torch.rand(4, 3, 512, 512).to(dev)
    const = torch.ones(4, 512, 4, 4).to(dev)

    m = MappingNetwork(z_dim=512).to(dev)
    g = Generator(channel_list=[512, 512, 512, 512, 256, 128, 64, 32]).to(dev)
    d = Discriminator(channel_list=[512, 512, 512, 512, 256, 128, 64, 32]).to(dev)
    alpha = 0.5

    w = m(z)
    w1, w2 = torch.split(w, 4, dim=0)
    # print("Generator")
    # for step in range(8):
    #     out = g(const, w1, w2, step=step, alpha=alpha)
    #     print(out.shape)
    print("Discriminator")
    out = d(img, step=7, alpha=alpha)
    print(out.shape)
