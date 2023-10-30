import torch
import torchvision

class VGGPerceptualLoss(torch.nn.Module):

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()

        vgg = torchvision.models.vgg16(pretrained=True)

        blocks = []
        blocks.append(vgg.features[:4].eval())
        blocks.append(vgg.features[4:9].eval())
        blocks.append(vgg.features[9:16].eval())
        blocks.append(vgg.features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False

        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, target):

        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)

        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)

        losses = []

        x = input
        y = target

        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            losses.append(torch.nn.functional.smooth_l1_loss(x, y))

        return losses

    def get_features(self, input):

        input = (input - self.mean) / self.std

        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)

        x = input

        for i, block in enumerate(self.blocks):
            x = block(x)

        return x