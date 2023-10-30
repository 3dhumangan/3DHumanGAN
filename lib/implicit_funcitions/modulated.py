import torch
import torch.nn as nn
import lib.components.pigan_layers as sin_layers


class COORDCONCATSIREN(nn.Module):

    def __init__(self, input_dim=2, latent_dim=100, hidden_dim=256, geo_feature_dim=88, output_dim=1, feature_dim=32, num_blocks=9, device=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.geo_feature_dim = geo_feature_dim
        self.output_dim = output_dim

        self.first_layer_coord = sin_layers.SineLayer(input_dim, hidden_dim)
        self.first_layer_mod = sin_layers.SineLayer(geo_feature_dim, hidden_dim)

        network = [sin_layers.FiLMLayer(2 * hidden_dim, hidden_dim)]
        for i in range(num_blocks - 1):
            network.append(sin_layers.FiLMLayer(hidden_dim, hidden_dim))
        self.network = nn.ModuleList(network)

        self.sigma_layer = nn.Linear(hidden_dim, 1)

        self.color_layer_sine = sin_layers.FiLMLayer(hidden_dim + 3, hidden_dim)
        self.color_layer_linear = nn.Linear(hidden_dim, 3)

        self.feature_layer_linear = nn.Linear(hidden_dim, feature_dim)

        self.network.apply(sin_layers.frequency_init(25))
        self.sigma_layer.apply(sin_layers.frequency_init(25))
        self.color_layer_sine.apply(sin_layers.frequency_init(25))
        self.color_layer_linear.apply(sin_layers.frequency_init(25))
        self.feature_layer_linear.apply(sin_layers.frequency_init(25))
        self.first_layer_coord.apply(sin_layers.first_layer_sine_init)
        self.first_layer_mod.apply(sin_layers.first_layer_sine_init)


    def forward(self, input, frequencies, phase_shifts, geo_feature, ray_directions, input_scaler=1., geo_feature_scaler=1., **kwargs):

        frequencies = frequencies * 15 + 30
        input = input * input_scaler
        geo_feature = geo_feature * geo_feature_scaler

        # print()
        # print_stats(input)

        unsqueeze = len(input.size()) < 3
        if unsqueeze:
            input = input.unsqueeze(1)
            geo_feature = geo_feature.unsqueeze(1)
            ray_directions = ray_directions.unsqueeze(1)

        input = self.first_layer_coord(input)
        geo_feature = self.first_layer_mod(geo_feature)
        x = torch.cat([input, geo_feature], dim=-1)

        for index, layer in enumerate(self.network):
            start = index * self.hidden_dim
            end = (index + 1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.sigma_layer(x)

        x = torch.cat([ray_directions, x], dim=-1)
        x = self.color_layer_sine(x, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])

        rgb = torch.sigmoid(self.color_layer_linear(x))
        x = self.feature_layer_linear(x)

        out = torch.cat([rgb, x, sigma], dim=-1)
        if unsqueeze: out = out.squeeze(1)

        return out