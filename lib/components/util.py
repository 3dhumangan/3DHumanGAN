import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def kaiming_leaky_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

def kaiming_linear_init(m):
    with torch.no_grad():
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, a=1., mode='fan_in', nonlinearity='leaky_relu')


class LatentPool (nn.Module):

    def __init__(self, pool_size, latent_dim):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros([pool_size, latent_dim]), requires_grad=True)

    def init(self, latents):
        with torch.no_grad():
            self.latents.copy_(latents)

    def forward(self, indices):
        return self.latents[indices]


def positional_encoding(x, dim, L=10):

    n_channels = x.shape[dim]

    repeater = [1] * len(x.shape)
    repeater[dim] = L

    frequencies = math.pi * torch.arange(1, L+1, device=x.device, dtype=x.dtype)
    frequencies = frequencies.view(repeater).repeat_interleave(n_channels, dim)

    x_sin = torch.sin(x.repeat(repeater) * frequencies)
    x_cos = torch.cos(x.repeat(repeater) * frequencies)

    return torch.cat([x_sin, x_cos], dim=dim)


def position_l1_loss(x, y, dim, L=10, beta=0.1):

    x_enc = positional_encoding(x, dim, L)
    y_enc = positional_encoding(y, dim, L)

    loss = F.smooth_l1_loss(x, y, beta=beta) + F.smooth_l1_loss(x_enc, y_enc, beta=beta)

    return 0.5 * loss


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


def apply_transformation(points, transformation):
    """
    Args (case #1):
        points: [batch_size, ..., 3]
        transformation: [batch_size, 4, 4]
    Args (case #2):
        points: [batch_size, ..., num_joints, 3]
        transformation: [batch_size, num_joints, 4, 4]
    """

    points_homo = F.pad(points, (0, 1), mode="constant", value=1.)
    points_homo = points_homo.unsqueeze(-1)
    points_transformed = torch.matmul(transformation, points_homo)
    points_transformed = points_transformed.squeeze(-1)[..., :3]

    # size = points.size()
    # points = points.view(size[0], list_prod(size[1:-1]), size[-1])
    # points_homo = F.pad(points, (0, 1), mode="constant", value=1.)
    # points_homo = points_homo.permute(0, 2, 1)
    # points_transformed = torch.bmm(transformation, points_homo).permute(0, 2, 1)[..., :3]
    # points_transformed = points_transformed.view(size)

    return points_transformed


def normalize_vecs(vectors: torch.Tensor) -> torch.Tensor:
    """
    Normalize vector lengths.
    """
    return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-12)


def perspective_projection(points, intrinsics, extrinsics):
    """
    Args:
        points: [batch_size, n_points, 3]
        intrinsics: [batch_size, 3, 3]
        extrinsics: [batch_size, 4, 4]
        T: [batch_size, 3]
    """

    points = apply_transformation(points, extrinsics.unsqueeze(1))
    projected = torch.matmul(intrinsics.unsqueeze(1), points.unsqueeze(-1)).squeeze(-1)
    projected[..., :2] = projected[..., :2] / (projected[..., 2:3] + 1e-12)
    return projected


def skeleton_uvd_to_heatmap(uv, resolution, depth=None, sigma=0.06):
    """
    Args:
        uv: shape: [B, N_joints, 2], range: [-1, 1]
        resolution: (width, height)
        depth: shape [B, N_joints], range: [-1, 1]

    Returns: [B, N_joints, height, width], range: [0, 1]
    """

    batch_size, num_joints, _ = uv.shape
    width, height = resolution

    horizontal_span = width / height
    grix_x, grid_y = torch.meshgrid(torch.linspace(-horizontal_span, horizontal_span, width, device=uv.device),
                                    torch.linspace(-1, 1, height, device=uv.device))

    grix_x = grix_x.T.view(1, 1, height, width).expand(batch_size, num_joints, height, width)
    grid_y = grid_y.T.view(1, 1, height, width).expand(batch_size, num_joints, height, width)

    dist_x = grix_x - uv[:, :, 0].view(batch_size, num_joints, 1, 1)
    dist_y = grid_y - uv[:, :, 1].view(batch_size, num_joints, 1, 1)

    dist_to_center = dist_x ** 2 + dist_y ** 2
    heatmaps = torch.exp(-0.5 * dist_to_center / sigma ** 2)

    if depth is not None:
        depth = torch.clamp(depth, -1., 1.)
        depth = depth.view(batch_size, num_joints, 1, 1).expand(batch_size, num_joints, height, width)
        heatmaps = torch.stack([heatmaps, depth], dim=2)

    return heatmaps


def visualize_heatmap(skeleton_uvd, resolution, depth_length):

    batch_size, n_joints, _ = skeleton_uvd.size()

    skeleton_depth = skeleton_uvd[..., 2:3] / (depth_length / 2.)
    heatmap = skeleton_uvd_to_heatmap(skeleton_uvd[..., :2], resolution)
    base_freq = 0.5 * math.pi
    red = torch.sin(base_freq * skeleton_depth) + 1.
    green = torch.sin(-base_freq * skeleton_depth) + 1.
    blue = 0.5 * torch.cos(base_freq * skeleton_depth) + 0.5
    heatmap = torch.stack([red, green, blue], dim=2).view(batch_size, n_joints, 3, 1, 1) * heatmap.unsqueeze(2)
    heatmap, _ = torch.max(heatmap, dim=1)

    return heatmap
