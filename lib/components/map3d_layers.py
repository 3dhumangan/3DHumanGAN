import math
import torch
import torch.nn as nn

import torch.distributed as dist
from lib.components.util import normalize_2nd_moment, kaiming_leaky_init


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

class SpatialStyleModLayer(nn.Module):
    def __repr__(self): return f"{self.__class__.__name__}({self.repr})"

    def __init__(self, in_channel, out_channel, kernel_size=1,
                 style_dim=None, demodulate=True, eps=1e-8, **kwargs):
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

        self.padding = kernel_size // 2

        assert kernel_size == 1
        self.weight = nn.Parameter(torch.randn(1, 1, in_channel, out_channel) * math.sqrt(2 / (1 + 0.2**2)) / math.sqrt(in_channel))

        self.bias = nn.Parameter(torch.zeros(1, 1, out_channel))

        self.affine = nn.Linear(style_dim, in_channel)
        torch.nn.init.kaiming_normal_(self.affine.weight, mode='fan_in', nonlinearity='linear')

        self.repr = f"in_channel={in_channel}, out_channel={out_channel}, kernel_size={kernel_size}, " \
                    f"style_dim={style_dim}, demodulate={demodulate}, "

    def forward(self, x, style):

        if len(style.shape) > 3:
            B, C, H, W = style.shape
            style = style.permute(0, 2, 3, 1).view(B, H*W, C)

        mod = self.affine(style) + 1

        # (1, 1, in, out) * (b, n_pix, in) -> (b, n_pix, in, out)
        weight = self.weight * mod.unsqueeze(-1)

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=2, keepdims=True) + self.eps)  # (b, n_pix, 1, out)
            weight = weight * demod  # (b, n_pix, in, out) * (b, n_pix, 1, out) -> (b, n_pix, in, out)

        # (b, n_pix, in) * (b, n_pix, in, out) -> (b, n_pix, out)
        out = (x.unsqueeze(-1) * weight).sum(dim=2)

        out = out + self.bias

        return out


class SynthesisBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, name_prefix):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.name_prefix = name_prefix

        self.mod1 = SpatialStyleModLayer(in_channel=in_dim, out_channel=out_dim, style_dim=style_dim)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.mod2 = SpatialStyleModLayer(in_channel=out_dim, out_channel=out_dim, style_dim=style_dim)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.skip = SkipLayer()


    def forward(self, x, style, skip=False):
        x_orig = x

        x = self.mod1(x, style)
        x = self.act1(x)

        x = self.mod2(x, style)
        out = self.act2(x)

        if skip and out.shape[-1] == x_orig.shape[-1]:
            out = self.skip(out, x_orig)
        return out


class SpatialAdaptiveBatchNorm2d (nn.Module):

    def __init__(self, num_features, momentum=0.05, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.momentum = momentum

        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, input):
        """
        x: B, C, H, W
        """

        if self.training:

            with torch.no_grad():

                var, mean = torch.var_mean(input, dim=[0, 2, 3], keepdim=True, unbiased=True)
                world_size = dist.get_world_size()

                dist.all_reduce(mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(var, op=dist.ReduceOp.SUM)

                self.running_mean += (mean / world_size - self.running_mean) * self.momentum
                self.running_var += (var / world_size - self.running_var) * self.momentum

            self.num_batches_tracked += 1

        output = (input - self.running_mean) * torch.rsqrt(self.running_var + self.eps)

        return output



class SPADE2d(nn.Module):

    def __init__(self, input_dim, feature_dim, normalization="instance_norm"):
        super().__init__()

        self.normalization = normalization
        if normalization == "instance_norm":
            self.first_norm = nn.InstanceNorm2d(input_dim)
        elif normalization == "batch_norm":
            self.first_norm = nn.SyncBatchNorm(input_dim)
        elif normalization == "adaptive_batch_norm":
            self.first_norm = SpatialAdaptiveBatchNorm2d(input_dim)
        else:
            self.first_norm = nn.Identity()

        hidden_dim = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)
        self.mlp_beta = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, x, feature_maps):

        normalized = self.first_norm(x)
        actv = self.mlp_shared(feature_maps)

        gamma = 1 + self.mlp_gamma(actv)

        if hasattr(self, "normalization") and self.normalization == "none":
            gamma = normalize_2nd_moment(gamma, dim=1)
            out = normalized * gamma
        else:
            beta = self.mlp_beta(actv)
            out = normalized * gamma + beta

        return out


class SPADEBlock(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, normalization="instance_norm"):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim

        self.conv_0 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.conv_1 = nn.Conv2d(out_dim, out_dim, kernel_size=1)

        self.conv_0 = nn.utils.spectral_norm(self.conv_0)
        self.conv_1 = nn.utils.spectral_norm(self.conv_1)

        self.spade_0 = SPADE2d(in_dim, style_dim, normalization)
        self.spade_1 = SPADE2d(out_dim, style_dim, normalization)

        self.activation = nn.LeakyReLU(0.2, inplace=True)

        self.skip = SkipLayer()

        self.apply(kaiming_leaky_init)


    def forward(self, x, style, skip=False):

        if len(style.shape) < 4:
            batch_size, _, style_dim = style.shape
            _, _, height, width = x.shape
            style = style.view(batch_size, style_dim, 1, 1).expand(batch_size, style_dim, height, width)

        x_orig = x

        x = self.spade_0(x, style)
        x = self.activation(x)
        x = self.conv_0(x)

        x = self.spade_1(x, style)
        x = self.activation(x)
        x = self.conv_1(x)

        if skip and x.shape[-1] == x_orig.shape[-1]:
            x = self.skip(x, x_orig)

        return x


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


class SynthesisStyleInput(nn.Module):

    def __init__(self, input_dim, latent_dim, output_dim, num_layers=1):

        super().__init__()

        self.latent_dim = latent_dim

        self.from_coords = nn.Sequential(nn.Conv2d(input_dim, latent_dim, kernel_size=1),
                                         SinAct())
        nn.init.uniform_(self.from_coords[0].weight, -math.sqrt(9 / input_dim), math.sqrt(9 / input_dim))

        network = [nn.Conv2d(latent_dim * 2, output_dim, kernel_size=1),
                   nn.LeakyReLU(0.2, inplace=True)]
        nn.init.kaiming_normal_(network[0].weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

        for i in range(1, num_layers - 1):
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


    def forward(self, coords, latent):

        B, _, H, W = coords.shape

        latent = normalize_2nd_moment(latent, dim=1)

        fourier_features = self.from_coords(coords)

        x = torch.cat([fourier_features, latent.view(B, self.latent_dim, 1, 1).expand(B, self.latent_dim, H, W)], dim=1)

        style = self.network(x)

        return style


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