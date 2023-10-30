import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lib.components.util import kaiming_leaky_init

def cips_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            torch.nn.init.normal_(m.weight, 0, 1 / math.sqrt(num_input))

def cips_bias_init(bias):
    def init_fn(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                torch.nn.init.normal_(m.weight, 0, 1 / math.sqrt(num_input))
                torch.nn.init.constant_(m.bias, bias)
    return init_fn


class L2Norm(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=self.dim, keepdim=True) + 1e-12)


class AttentionLinear(nn.Module):

    def __init__(self, in_dim, modulation_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim)
        self.linear.apply(kaiming_leaky_init)

        self.modulation = nn.Linear(modulation_dim, in_dim)
        self.modulation.apply(cips_bias_init(0.))

    def forward(self, x, modulation=None):

        if modulation is not None:

            modulation = self.modulation(modulation)
            modulation = torch.softmax(modulation, dim=-1)

            # print("modulation: min={}, mean={}, max={}, std={}, abs_mean={}".format(
            #     modulation.min(), modulation.mean(), modulation.max(), modulation.std(), torch.abs(modulation).mean()))

            out = self.linear(x * modulation)

        else:

            out = self.linear(x)

        return out


class ModulatedLinear(nn.Module):

    def __init__(self, in_dim, modulation_dim, out_dim, demodulate=True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.demodulate = demodulate

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        with torch.no_grad():
            torch.nn.init.kaiming_normal_(self.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.modulation = nn.Linear(modulation_dim, in_dim)
        self.modulation.apply(cips_bias_init(1.))

    def forward(self, x, modulation=None):

        weight = self.weight

        if modulation is not None:

            modulation = self.modulation(modulation)
            weight = weight.view(1, 1, self.out_dim, self.in_dim) * modulation.unsqueeze(-2)

            if self.demodulate:
                with torch.no_grad():
                    demod = torch.rsqrt(weight.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
                    weight = weight * demod

            # print("modulation: min={}, mean={}, max={}, std={}, abs_mean={}".format(
            #     modulation.min(), modulation.mean(), modulation.max(), modulation.std(), torch.abs(modulation).mean()))

            out = torch.matmul(weight, x.unsqueeze(-1)).squeeze(-1)
            out = out + self.bias

        else:
            out = F.linear(x, weight, bias=self.bias)

        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, name_prefix, ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.name_prefix = name_prefix

        self.net = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=out_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(in_features=out_dim, out_features=out_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net.apply(kaiming_leaky_init)

        pass

    def forward(self, x, *args, **kwargs):
        out = self.net(x)
        return out

    def __repr__(self):
        repr = f"{self.__class__.__name__}(in_dim={self.in_dim}, " \
                     f"out_dim={self.out_dim})"
        return repr


class SinAct(nn.Module):
    def __init__(self, ):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class SkipLayer(nn.Module):
    def __init__(self, ):
        super(SkipLayer, self).__init__()

    def forward(self, x0, x1):
        # out = (x0 + x1) / math.pi
        out = (x0 + x1)
        return out


class StyleModLayer(nn.Module):
    def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

    def __init__(self, in_channel, out_channel, kernel_size=1,
                 style_dim=None, demodulate=True, use_group_conv=True, eps=1e-8, **kwargs):
        """

        :param in_channel:
        :param out_channel:
        :param kernel_size:
        :param style_dim: =in_channel
        :param demodulate:
        """
        super().__init__()

        self.eps = eps
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.style_dim = style_dim
        self.demodulate = demodulate
        self.use_group_conv = use_group_conv

        self.padding = kernel_size // 2

        if use_group_conv:
            self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
            torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')
        else:
            assert kernel_size == 1
            self.weight = nn.Parameter(torch.randn(1, in_channel, out_channel))
            torch.nn.init.kaiming_normal_(self.weight[0], a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        self.bias = nn.Parameter(torch.zeros(1, out_channel))

        self.geo_feature = nn.Linear(style_dim, in_channel)
        self.geo_feature.apply(kaiming_leaky_init)

        self.repr = f"in_channel={in_channel}, out_channel={out_channel}, kernel_size={kernel_size}, " \
                    f"style_dim={style_dim}, demodulate={demodulate}, " \
                    f"use_group_conv={use_group_conv}"

    def forward_bmm(self, x, style, weight):
        """

        :param input: (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, "b c -> b 1 c")
        elif x.dim() == 3:
            input = x
        else:
            raise Exception("wrong input size")

        batch, N, in_channel = input.shape

        style = self.geo_feature(style)
        style = style.view(-1, in_channel, 1)

        # (1, in, out) * (b in 1) -> (b, in, out)
        weight = weight * (style + 1)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([1, ]) + self.eps)  # (b, out)
            weight = weight * demod.view(batch, 1, self.out_channel)  # (b, in, out) * (b, 1, out) -> (b, in, out)
        # (b, n, in) * (b, in, out) -> (b, n, out)
        out = torch.bmm(input, weight)

        if x.dim() == 2:
            out = rearrange(out, "b 1 c -> b c")
        elif x.dim() == 3:
            pass

        out = out + self.bias.view(1, 1, self.out_channel)

        return out

    def forward_group_conv(self, x, style):
        """

        :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        assert x.shape[0] == style.shape[0]
        if x.dim() == 2:
            input = rearrange(x, "b c -> b c 1 1")
        elif x.dim() == 3:
            input = rearrange(x, "b n c -> b c n 1")
        elif x.dim() == 4:
            input = x
        else:
            assert 0

        batch, in_channel, height, width = input.shape

        style = self.geo_feature(style).view(-1, 1, in_channel, 1, 1)
        style = style + 1.

        # (1, out, in, ks, ks) * (b, 1, in, 1, 1) -> (b, out, in, ks, ks)
        weight = self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + self.eps) # (b, out)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1) # (b, out, in, ks, ks) * (b, out, 1, 1, 1)
        # (b*out, in, ks, ks)
        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)
        # (1, b*in, h, w)
        input = input.reshape(1, batch * in_channel, height, width)
        out = F.conv2d(input, weight, padding=self.padding, groups=batch)
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        if x.dim() == 2:
            out = rearrange(out, "b c 1 1 -> b c")
        elif x.dim() == 3:
            out = rearrange(out, "b c n 1 -> b n c")

        out = out + self.bias.view(1, self.out_channel, 1, 1)

        return out

    def forward(self, x, style):
        """

        :param input: (b, in_c, h, w), (b, in_c), (b, n, in_c)
        :param style: (b, in_c)
        :return:
        """
        if self.use_group_conv:
            out = self.forward_group_conv(x=x, style=style)
        else:
            out = self.forward_bmm(x=x, style=style, weight=self.weight)

        out = out

        return out


class SynthesisInput(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers=1):

        super().__init__()

        network = [nn.Conv2d(input_dim, output_dim, kernel_size=1),
                   SinAct()]
        nn.init.uniform_(network[0].weight, -math.sqrt(9 / input_dim), math.sqrt(9 / input_dim))

        for i in range(1, num_layers):
            layer = nn.Conv2d(output_dim, output_dim, kernel_size=1)
            nn.init.kaiming_normal_(layer.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            network.append(layer)
            network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network = nn.Sequential(*network)


    def get_2d_coords(self, batch_size, height, width, dtype, device):

        i_coords = torch.linspace(-1, 1, height, dtype=dtype, device=device)
        j_coords = torch.linspace(-1, 1, width, dtype=dtype, device=device)

        i_coords, j_coords = torch.meshgrid(i_coords, j_coords)
        coords = torch.stack([i_coords, j_coords], dim=0)
        coords = coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        return coords

    def forward(self, coords):

        fourier_features = self.network(coords)

        return fourier_features


class ToRGB(nn.Module):

    def __init__(self, in_dim, dim_rgb=3, use_conv=True):
        super().__init__()
        self.in_dim = in_dim
        self.dim_rgb = dim_rgb

        if use_conv:
            self.linear = nn.Conv2d(in_dim, dim_rgb, 1)
        else:
            self.linear = nn.Linear(in_dim, dim_rgb)
        pass

        with torch.no_grad():
            self.linear.weight *= 0.25

    def forward(self, input, rgb=None):

        out = self.linear(input)

        if rgb is not None:
            out = out + rgb
        return out