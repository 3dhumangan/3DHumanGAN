import numpy as np
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
from lib.data.utils import print_stats


class Sine(nn.Module):
    """Sine Activation Function."""

    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.sin(30. * x)

def sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_film_sine_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def frequency_init(freq):
    def init_fn(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                num_input = m.weight.size(-1)
                m.weight.uniform_(-np.sqrt(6 / num_input) / freq, np.sqrt(6 / num_input) / freq)
    return init_fn


def kaiming_leaky_bias_init(bias):
    def init_fn(m):
        with torch.no_grad():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                torch.nn.init.constant_(m.bias, bias)
    return init_fn


class SineLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, omega_0=30.):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.omega_0 = omega_0

    def forward(self, x):
        x = self.layer(x)
        return torch.sin(self.omega_0 * x)


class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, freq, phase_shift):

        x = self.layer(x)

        if len(x.shape) > len(freq.shape):
            freq = freq.unsqueeze(1).expand_as(x)
            phase_shift = phase_shift.unsqueeze(1).expand_as(x)

        return torch.sin(freq * x + phase_shift)

class ModLayer(nn.Module):

    def __init__(self, input_dim, modulation_dim, output_dim, init_freq=25, is_first=False, is_last=False):
        super().__init__()

        self.output_dim = output_dim
        self.layer = nn.Linear(input_dim, output_dim)
        self.freq_and_phase = nn.Linear(modulation_dim, output_dim * 2)

        self.layer.apply(frequency_init(init_freq) if not is_first else first_layer_film_sine_init)

        torch.nn.init.kaiming_normal_(self.freq_and_phase.weight, mode='fan_in', nonlinearity='linear')
        with torch.no_grad(): self.freq_and_phase.weight *= 0.25

    def forward(self, x, style):

        x = self.layer(x)

        style = F.leaky_relu(style, negative_slope=0.2)
        freq_and_phase = self.freq_and_phase(style)
        freq, phase = torch.split(freq_and_phase, self.output_dim, dim=-1)

        # print("freq min={:.3f}, mean={:.3f}, max={:.3f}, std={:.3f}".format(
        #     freq.min(), freq.mean(), freq.max(), freq.std()))
        # print("phase min={:.3f}, mean={:.3f}, max={:.3f}, std={:.3f}".format(
        #     phase.min(), phase.mean(), phase.max(), phase.std()))

        freq = freq * 15 + 30
        return torch.sin(freq * x + phase)