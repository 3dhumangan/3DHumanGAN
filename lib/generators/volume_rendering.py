import time
from functools import partial

import math
import numpy as np
import torch
import torch.nn.functional as F
import random
from lib.components.util import normalize_vecs


def ray_integration(input, z_vals, device, noise_std=0.5, last_back=False, white_back=False, clamp_mode=None, fill_mode=None):
    """Performs NeRF volumetric rendering."""

    # [ batch_size, num_rays, num_steps, num_channels ]
    features = input[..., :-1]
    sigmas = input[..., -1:]

    # [ batch_size, num_rays, num_steps, num_channels ]
    deltas = z_vals[:, :, 1:] - z_vals[:, :, :-1]
    delta_inf = 1e9 * torch.ones_like(deltas[:, :, :1])
    deltas = torch.cat([deltas, delta_inf], -2)

    noise = torch.randn(sigmas.shape, device=device) * noise_std

    if clamp_mode == 'softplus':
        alphas = 1 - torch.exp(-deltas * (F.softplus(sigmas + noise)))
    elif clamp_mode == 'relu':
        alphas = 1 - torch.exp(-deltas * (F.relu(sigmas + noise)))
    else:
        raise Exception("Need to choose clamp mode")

    # alphas_shifted (~termination prob): step[0] = 1, step[1:] = exp(-sigmas)
    alphas_shifted = torch.cat([torch.ones_like(alphas[:, :, :1]), 1-alphas + 1e-12], -2)
    weights = alphas * torch.cumprod(alphas_shifted, -2)[:, :, :-1]
    weights_sum = weights.sum(2)

    if last_back: # whether or not to render the last sampled step as background
        weights[:, :, -1] += (1 - weights_sum)
        features_final = torch.sum(weights * features, -2)
        depth_final = torch.sum(weights * z_vals, -2)
    else:
        features_final = torch.sum(weights * features, -2)
        weights_depth = weights.clone()
        weights_depth[:, :, -1] += (1 - weights_sum)
        depth_final = torch.sum(weights_depth * z_vals, -2)

    if white_back:
        features_final = features_final + 1 - weights_sum

    if fill_mode == 'debug':
        features_final[weights_sum.squeeze(-1) < 0.9] = torch.tensor([1., 0, 0], device=features_final.device)
    elif fill_mode == 'weight':
        features_final = weights_sum.expand_as(features_final)

    return features_final, depth_final, weights


def get_initial_rays_trig(n, num_steps, device, fov, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""

    W, H = resolution
    # Create full screen NDC (-1 to +1) coords [x, y, 0, 1].
    horizontal_span = W / H
    x, y = torch.meshgrid(torch.linspace(-horizontal_span, horizontal_span, W, device=device),
                          torch.linspace(-1, 1, H, device=device))
    x = x.T.flatten()
    y = y.T.flatten()

    fov = math.pi * (fov / 180)
    focal = 1. / np.tan(fov / 2)
    z = torch.ones_like(x, device=device) * focal

    rays_d_cam = normalize_vecs(torch.stack([x, y, z], -1))

    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device).reshape(1, num_steps, 1).repeat(W*H, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, num_steps, 1) * z_vals

    points = torch.stack(n*[points])
    z_vals = torch.stack(n*[z_vals])
    rays_d_cam = torch.stack(n*[rays_d_cam]).to(device)

    return points, z_vals, rays_d_cam


def get_initial_rays_weak_perspective(focals, scales, num_steps, device, resolution, ray_start, ray_end):
    """Returns sample points, z_vals, and ray directions in camera space."""

    batch_size = focals.size(0)
    W, H = resolution

    horizontal_span = W / H
    x, y = torch.meshgrid(torch.linspace(-horizontal_span, horizontal_span, W, device=device),
                          torch.linspace(-1, 1, H, device=device))

    x = x.T.flatten().unsqueeze(0).expand(batch_size, H * W)
    y = y.T.flatten().unsqueeze(0).expand(batch_size, H * W)
    z = torch.ones_like(x, device=device) * focals.unsqueeze(1)
    # [ batch_size, H * W ]

    xyz = torch.stack([x, y, z], dim=-1)
    rays_d_cam = normalize_vecs(xyz)  # [ batch_size, H * W, 3 ]

    z_vals = torch.linspace(ray_start, ray_end, num_steps, device=device)
    z_vals = z_vals.reshape(1, 1, num_steps, 1).expand(batch_size, H * W, num_steps, 1)
    z_vals = z_vals + (focals / scales).view(batch_size, 1, 1, 1)

    points = rays_d_cam.unsqueeze(2).expand(batch_size, H * W, num_steps, 3) * z_vals

    return points, z_vals, rays_d_cam


def expand_ray_directions(ray_directions, num_steps):

    batch_size, num_rays, _ = ray_directions.size()

    ray_directions_exp = ray_directions.unsqueeze(-2)
    ray_directions_exp = ray_directions_exp.expand(-1, -1, num_steps, -1)
    ray_directions_exp = ray_directions_exp.reshape(batch_size, num_rays * num_steps, 3)

    return ray_directions_exp


def perturb_points(points, z_vals, ray_directions, device):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=device)-0.5) * distance_between_points
    z_vals = z_vals + offset

    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def transform_sampled_points(points, z_vals, ray_directions, device, cam2world_matrix=None, h_stddev=1, v_stddev=1, h_mean=math.pi * 0.5, v_mean=math.pi * 0.5, mode='normal', cano_matrices=None, perturb=True):
    """Samples a camera position and maps points in camera space to world space."""

    n, num_rays, num_steps, channels = points.shape

    if perturb:
        points, z_vals = perturb_points(points, z_vals, ray_directions, device)

    camera_origin, pitch, yaw = sample_camera_positions(n=points.shape[0], r=1, horizontal_stddev=h_stddev,
                                                        vertical_stddev=v_stddev, horizontal_mean=h_mean,
                                                        vertical_mean=v_mean, device=device, mode=mode)
    forward_vector = normalize_vecs(-camera_origin)

    if cam2world_matrix is None:
        cam2world_matrix = create_cam2world_matrix(forward_vector, camera_origin, device=device)

    world2cam_matrix = torch.inverse(cam2world_matrix.float())

    points_homogeneous = F.pad(points, (0, 1), mode="constant", value=1.)

    # should be n x 4 x 4 , n x r^2 x num_steps x 4
    points_homogeneous = points_homogeneous.reshape(n, -1, 4).permute(0,2,1)
    transformed_points = torch.bmm(cam2world_matrix, points_homogeneous)
    if cano_matrices is not None: transformed_points = torch.bmm(cano_matrices, transformed_points)
    transformed_points = transformed_points.permute(0, 2, 1).reshape(n, num_rays, num_steps, 4)

    ray_directions = ray_directions.reshape(n, -1, 3).permute(0,2,1)
    transformed_ray_directions = torch.bmm(cam2world_matrix[..., :3, :3], ray_directions)
    if cano_matrices is not None: transformed_ray_directions = torch.bmm(cano_matrices[..., :3, :3], transformed_ray_directions)
    transformed_ray_directions = transformed_ray_directions.permute(0, 2, 1).reshape(n, num_rays, 3)

    homogeneous_origins = torch.zeros((n, 4, num_rays), device=device)
    homogeneous_origins[:, 3, :] = 1
    transformed_ray_origins = torch.bmm(cam2world_matrix, homogeneous_origins)
    if cano_matrices is not None: transformed_ray_origins = torch.bmm(cano_matrices, transformed_ray_origins)
    transformed_ray_origins = transformed_ray_origins.permute(0, 2, 1).reshape(n, num_rays, 4)[..., :3]

    return transformed_points[..., :3], z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw, world2cam_matrix


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def sample_camera_positions(device, n=1, r=1, horizontal_stddev=1, vertical_stddev=1, horizontal_mean=math.pi*0.5, vertical_mean=math.pi*0.5, mode='normal'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """

    if mode == 'uniform':
        theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev + horizontal_mean
        phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev + vertical_mean

    elif mode == 'normal' or mode == 'gaussian':
        theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
        phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'hybrid':
        if random.random() < 0.5:
            theta = (torch.rand((n, 1), device=device) - 0.5) * 2 * horizontal_stddev * 2 + horizontal_mean
            phi = (torch.rand((n, 1), device=device) - 0.5) * 2 * vertical_stddev * 2 + vertical_mean
        else:
            theta = torch.randn((n, 1), device=device) * horizontal_stddev + horizontal_mean
            phi = torch.randn((n, 1), device=device) * vertical_stddev + vertical_mean

    elif mode == 'truncated_gaussian':
        theta = truncated_normal_(torch.zeros((n, 1), device=device)) * horizontal_stddev + horizontal_mean
        phi = truncated_normal_(torch.zeros((n, 1), device=device)) * vertical_stddev + vertical_mean

    elif mode == 'spherical_uniform':
        theta = 2. * torch.rand((n, 1), device=device) - 1. # [-1, 1]
        theta = theta * horizontal_stddev + horizontal_mean
        v_stddev, v_mean = vertical_stddev / math.pi, vertical_mean / math.pi
        v = 2. * torch.rand((n, 1), device=device) - 1.
        v = v * v_stddev + v_mean
        v = torch.clamp(v, 1e-5, 1 - 1e-5)
        phi = torch.arccos(1 - 2 * v)

    else:
        # Just use the mean.
        theta = torch.ones((n, 1), device=device, dtype=torch.float) * horizontal_mean
        phi = torch.ones((n, 1), device=device, dtype=torch.float) * vertical_mean

    phi = torch.clamp(phi, 1e-5, math.pi - 1e-5)

    output_points = torch.zeros((n, 3), device=device)
    output_points[:, 0:1] = r*torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = r*torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = r*torch.cos(phi)

    return output_points, phi, theta

def create_cam2world_matrix(forward_vector, origin, device=None):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""

    forward_vector = normalize_vecs(forward_vector)
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=device).expand_as(forward_vector)

    left_vector = normalize_vecs(torch.cross(up_vector, forward_vector, dim=-1))

    up_vector = normalize_vecs(torch.cross(forward_vector, left_vector, dim=-1))

    rotation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((left_vector, up_vector, forward_vector), axis=-1)

    translation_matrix = torch.eye(4, device=device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin

    cam2world = translation_matrix @ rotation_matrix

    return cam2world



def create_world2cam_matrix(forward_vector, origin, device):
    """Takes in the direction the camera is pointing and the camera origin and returns a world2cam matrix."""
    cam2world = create_cam2world_matrix(forward_vector, origin, device=device)
    world2cam = torch.inverse(cam2world)
    return world2cam


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    Source: https://github.com/kwea123/nerf_pl/blob/master/models/rendering.py
    """

    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / torch.sum(weights, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1)
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1).view(N_rays, 2*N_importance)
    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples
