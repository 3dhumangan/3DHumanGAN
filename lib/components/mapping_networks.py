import math
import torch
import torch.nn as nn
import numpy as np

import lib.components.util as util
from lib.components.ops import bias_act
import lib.components.pigan_layers as pigan_layers
import lib.components.cips_layers as cips_layers
from lib.components.nv_misc import assert_shape


class MappingNetwork(nn.Module):

    def __init__(self, latent_dim, map_hidden_dim, map_output_dim):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(latent_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(util.kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):

        z = util.normalize_2nd_moment(z.to(torch.float32))

        freq_phase = self.network(z)
        freq = freq_phase[..., :freq_phase.shape[-1]//2]
        phase = freq_phase[..., freq_phase.shape[-1]//2:]

        return freq, phase


class TwoPartMappingNetwork_(nn.Module):

    def __init__(self, latent_dim, map_hidden_dim, implicit_output_dim, superres_output_dim, trunk_layers=4, branch_layers=4, normalize=False):
        super().__init__()

        trunk_network = [nn.Linear(latent_dim, map_hidden_dim),
                         nn.LeakyReLU(0.2, inplace=True)]

        for i in range(trunk_layers):
            trunk_network.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            trunk_network.append(nn.LeakyReLU(0.2, inplace=True))

        self.trunk_network = nn.Sequential(*trunk_network)
        self.trunk_network.apply(util.kaiming_leaky_init)

        implicit_branch = []
        for i in range(branch_layers-1):
            implicit_branch.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            implicit_branch.append(nn.LeakyReLU(0.2, inplace=True))
        implicit_branch.append(nn.Linear(map_hidden_dim, implicit_output_dim))
        self.implicit_branch = nn.Sequential(*implicit_branch)
        self.implicit_branch.apply(util.kaiming_leaky_init)
        with torch.no_grad():
            self.implicit_branch[-1].weight *= 0.25

        superres_branch = []
        for i in range(branch_layers - 1):
            superres_branch.append(nn.Linear(map_hidden_dim, map_hidden_dim))
            superres_branch.append(nn.LeakyReLU(0.2, inplace=True))
        superres_branch.append(nn.Linear(map_hidden_dim, superres_output_dim))
        self.superres_branch = nn.Sequential(*superres_branch)
        self.superres_branch.apply(pigan_layers.kaiming_linear_init)

        self.normalize = normalize

    def forward(self, z):

        if self.normalize:
            z = util.normalize_2nd_moment(z, dim=1)

        x = self.trunk_network(z)

        implicit_styles = self.implicit_branch(x)
        superres_styles = self.superres_branch(x)

        return implicit_styles, superres_styles


class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x


class TwoPartMappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        implicit_dim,
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        trunk_layers    = 6,        # Number of mapping layers.
        branch_layers   = 2,
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.implicit_dim = implicit_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.trunk_layers = trunk_layers
        self.branch_layers = branch_layers

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        trunk_channels = [z_dim + embed_features] + [layer_features] * trunk_layers
        implicit_channels = [layer_features] * branch_layers + [implicit_dim]
        superres_channels = [layer_features] * branch_layers + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(trunk_layers):
            in_features = trunk_channels[idx]
            out_features = trunk_channels[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'trunk{idx}', layer)

        for idx in range(branch_layers):
            in_features = implicit_channels[idx]
            out_features = implicit_channels[idx+1]
            layer = FullyConnectedLayer(
                in_features, out_features,
                activation='linear' if idx==branch_layers-1 else activation,
                lr_multiplier=lr_multiplier)
            setattr(self, f'implicit{idx}', layer)
        getattr(self, f'implicit{branch_layers-1}').weight_gain *= 0.2

        for idx in range(branch_layers):
            in_features = superres_channels[idx]
            out_features = superres_channels[idx+1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'superres{idx}', layer)


    def forward(self, z, c=None, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None

        if self.z_dim > 0:
            assert_shape(z, [None, self.z_dim])
            x = util.normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            assert_shape(c, [None, self.c_dim])
            y = util.normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.trunk_layers):
            layer = getattr(self, f'trunk{idx}')
            x = layer(x)

        x_implicit = x
        for idx in range(self.branch_layers):
            layer = getattr(self, f'implicit{idx}')
            x_implicit = layer(x_implicit)

        x_superres = x
        for idx in range(self.branch_layers):
            layer = getattr(self, f'superres{idx}')
            x_superres = layer(x_superres)

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x_superres = x_superres.unsqueeze(1).repeat([1, self.num_ws, 1])

        return x_implicit, x_superres
