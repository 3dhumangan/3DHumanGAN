import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):

    def __init__(self, fin, fout, up_or_down, first=False, **kwargs):
        super().__init__()

        # Attributes
        self.up_or_down = up_or_down
        self.first = first
        self.learned_shortcut = (fin != fout)
        fmiddle = fout

        norm_layer = (lambda x: x) if kwargs.get("disable_spectral_norm", False) else torch.nn.utils.spectral_norm

        if first:
            self.conv1 = norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1))
        else:
            if self.up_or_down > 0:
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2, False),
                    nn.Upsample(scale_factor=2),
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))
            else:
                self.conv1 = nn.Sequential(
                    nn.LeakyReLU(0.2, False),
                    norm_layer(nn.Conv2d(fin, fmiddle, 3, 1, 1)))

        self.conv2 = nn.Sequential(
            nn.LeakyReLU(0.2, False),
            norm_layer(nn.Conv2d(fmiddle, fout, 3, 1, 1)))

        if self.learned_shortcut:
            self.conv_s = norm_layer(nn.Conv2d(fin, fout, 1, 1, 0))

        if up_or_down > 0:
            self.sampling = nn.Upsample(scale_factor=2)
        elif up_or_down < 0:
            self.sampling = nn.AvgPool2d(2)
        else:
            self.sampling = nn.Sequential()

    def forward(self, x):

        x_s = self.shortcut(x)
        dx = self.conv1(x)
        dx = self.conv2(dx)
        if self.up_or_down < 0:
            dx = self.sampling(dx)
        out = x_s + dx
        return out

    def shortcut(self, x):
        if self.first:
            if self.up_or_down < 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            x_s = x
        else:
            if self.up_or_down > 0:
                x = self.sampling(x)
            if self.learned_shortcut:
                x = self.conv_s(x)
            if self.up_or_down < 0:
                x = self.sampling(x)
            x_s = x
        return x_s


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv2d') != -1:
        if hasattr(m, "weight"):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class UNetDiscriminator(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        self.epoch = 0
        self.step = 0

        self.semantic_dim = kwargs.get("semantic_dim", 0)
        self.label_dim = kwargs.get("label_dim", 0)
        self.latent_dim = kwargs["latent_dim"]
        self.output_dim = self.semantic_dim + self.label_dim

        self.num_blocks = kwargs.get("discriminator_blocks", 6)
        self.num_blocks = min(self.num_blocks, int(math.log2(max(kwargs["gen_height"], kwargs["gen_width"]))) - 1)

        input_channel = 6 if kwargs.get("dual_discrimination", False) else 3

        self.channels = [input_channel, 128, 128, 256, 256, 512, 512, 512, 512]
        self.body_up   = nn.ModuleList([])
        self.body_down = nn.ModuleList([])

        # encoder part
        for i in range(self.num_blocks):
            self.body_down.append(ResBlock(self.channels[i], self.channels[i + 1], -1, first=(i == 0), **kwargs))

        # decoder part
        self.body_up.append(ResBlock(self.channels[self.num_blocks], self.channels[self.num_blocks-1], 1, **kwargs))
        for i in range(1, self.num_blocks-1):
            self.body_up.append(ResBlock(2 * self.channels[self.num_blocks - i], self.channels[self.num_blocks - i - 1], 1, **kwargs))
        self.body_up.append(ResBlock(2 * self.channels[1], 64, 1, **kwargs))

        self.layer_up_last = nn.Conv2d(64, 1, 1, 1, 0)
        self.output_layer = nn.Conv2d(64, self.output_dim, 1, 1)

        downsample = 2 ** self.num_blocks
        self.latent_layer = nn.Conv2d(self.channels[self.num_blocks], self.latent_dim, (kwargs["gen_height"] // downsample, kwargs["gen_width"] // downsample))

        self.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.output_layer.weight *= 0.25


    def forward(self, images, conditions, alpha, **kwargs):

        x = images

        # encoder

        encoder_res = list()
        for i in range(len(self.body_down)):
            x = self.body_down[i](x)
            encoder_res.append(x)

        if min(x.shape[2:4]) > 1:
            latents = self.latent_layer(x).view(x.shape[0], self.latent_dim)
        else:
            latents = torch.zeros([x.shape[0], self.latent_dim], dtype=x.dtype, device=x.device)

        # decoder

        x = self.body_up[0](x)
        for i in range(1, len(self.body_up)):
            x = self.body_up[i](torch.cat((encoder_res[-i-1], x), dim=1))

        prediction = self.layer_up_last(x)
        x = self.output_layer(x)

        # print_stats(prediction.unsqueeze(-1), "prediction")
        # print_stats(semantics.unsqueeze(-1), "semantics")

        outputs = {"prediction": prediction,
                   "latents": latents,
                   "segments": x[:, self.semantic_dim:]}

        if self.semantic_dim > 0:
            outputs.update({"semantics": x[:, :self.semantic_dim],})

        return outputs


if __name__ == "__main__":

    D = UNetDiscriminator(output_dim=3, semantic_dim=3, latent_dim=512, gen_height=256, gen_width=128).cuda()
    print(sum(p.numel() for p in D.parameters()))
    images = torch.randn([1, 3, 64, 32]).cuda()
    disc_out = D.forward(images, None, 0.5)

    print(disc_out["prediction"].shape)
    print(disc_out["semantics"].shape)