import sys, os

import imageio

sys.path.insert(0, os.getcwd())

import argparse
import math
import numpy as np
import os

import torch

from tqdm import tqdm

import configs
import lib.generators
from lib.data import get_dataset, get_dataset_distributed, get_preprocessor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def generate_frames(generator, preprocessor, config, seed, conditions, n_angles, angle_range_h, angle_range_v, back_and_forth):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    z = torch.randn((1, config['latent_dim']), device=device)
    z = z.repeat_interleave(n_angles, dim=0).to(generator.device)
    # root_rotation = Rotation.from_matrix(conditions["root_rotation"][0].numpy()).as_euler("xyz")
    conditions = {k: v.repeat_interleave(n_angles, dim=0).to(generator.device) for k, v in conditions.items()}
    # print(root_rotation)

    if back_and_forth:
        angles_h = torch.linspace(-np.pi, np.pi, n_angles).to(generator.device)
        angles_v = torch.linspace(-np.pi, np.pi, n_angles).to(generator.device)
        angles_h = angle_range_h * torch.sin(angles_h).unsqueeze(-1)
        angles_v = angle_range_v * torch.cos(angles_v).unsqueeze(-1)
    else:
        angles_h = torch.linspace(-angle_range_h, angle_range_h, n_angles).to(generator.device).unsqueeze(-1)
        angles_v = torch.linspace(-angle_range_v, angle_range_v, n_angles).to(generator.device).unsqueeze(-1)
    angles_r = torch.zeros_like(angles_h)

    frames = torch.zeros(n_angles, 3, config["gen_height"], config["gen_width"]).float().to(generator.device)
    rasterized_semantics = torch.zeros(n_angles, 3, config["gen_height"], config["gen_width"]).float().to(generator.device)

    for i in range(n_angles):

        subset_conditions = {k: v[i:i+1] for k, v in conditions.items()}
        subset_conditions = preprocessor.forward_with_rotation(subset_conditions, angles_h[i:i+1], angles_v[i:i+1], angles_r[i:i+1], **config)

        smpl = torch.clamp(subset_conditions["rasterized_semantics"], -1, 1)
        bg_mask = torch.all(smpl==0, dim=1, keepdim=True)
        smpl[bg_mask.repeat(1, 3, 1, 1)] = 1
        rasterized_semantics[i:i + 1] = smpl

        gen_outputs_uncond = generator.staged_forward(z[i:i+1], subset_conditions, **config)
        frames[i:i+1] = torch.clamp(gen_outputs_uncond["rgbs"], -1, 1)

    frames = frames * 0.5 + 0.5
    frames = torch.clamp(frames * 255, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    rasterized_semantics = rasterized_semantics * 0.5 + 0.5
    rasterized_semantics = torch.clamp(rasterized_semantics * 255, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()

    return frames, rasterized_semantics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='MAP3DBN')
    parser.add_argument('--tune', type=str, default='')
    parser.add_argument('--variant', type=int, default=0)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(1, 10)))
    parser.add_argument('--dataroot', type=str, default='./datasets/shhq_example_dataset')
    parser.add_argument('--dataset_length', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='results/sample_from_generator')
    parser.add_argument('--postfix', type=str, default="")
    parser.add_argument('--lock_view_dependence', default=None)
    parser.add_argument('--n_angles', type=int, default=40)
    parser.add_argument('--back_and_forth', default=False, action="store_true")
    parser.add_argument('--save', type=str, default="mp4", choices=["mp4", "png", "gif"])
    parser.add_argument('--stitch', default=False, action="store_true")
    opt = parser.parse_args()

    # get config

    config = configs.get_config(opt)
    config = {key: value for key, value in config.items() if type(key) is str}

    config['truncation_psi'] = 0.7
    config['v_stddev'] = 0
    config['h_stddev'] = 0
    if opt.lock_view_dependence is not None: config['lock_view_dependence'] = opt.lock_view_dependence
    config['last_back'] = config.get('eval_last_back', False)
    config['nerf_noise'] = 0

    # create output directory

    output_dir = os.path.join(opt.output_dir, config['name'] + opt.postfix)
    os.makedirs(output_dir, exist_ok=True)

    # load model
    checkpoint = torch.load(opt.checkpoint)

    if isinstance(checkpoint, dict):
        generator = getattr(lib.generators, config['generator'])(**config).to(device)
        generator.load_state_dict(checkpoint)
    else:
        generator = torch.load(opt.checkpoint, map_location=device)
        ema_file = opt.path.split('generator')[0] + 'ema.pth'
        ema = torch.load(ema_file, map_location=device)
        ema.copy_to(generator.parameters())

    generator.set_device(device)
    generator.eval()

    # get data loader
    config['dataroot'] = opt.dataroot
    config['dataset_length'] = opt.dataset_length

    dataloader, _ = get_dataset(config['dataset'], inference=True, **config)
    dataiter = iter(dataloader)
    preprocessor = get_preprocessor(dataloader, config).to(device)
    preprocessor.device = device

    # generate
    for seed in tqdm(opt.seeds):

        data = next(dataiter)
        frames, rasterized_semantics = generate_frames(generator, preprocessor, config, seed, data, opt.n_angles, math.pi / 6, 0, opt.back_and_forth)

        if opt.stitch:
            for i in range(len(rasterized_semantics)):
                frames[i] = np.concatenate([frames[i], rasterized_semantics[i]], axis=0)

        if opt.save == "gif":
            imageio.mimwrite(os.path.join(output_dir, f"{seed:03d}_uncond.gif"), frames, fps=10)
            imageio.mimwrite(os.path.join(output_dir, f"{seed:03d}_smpl.gif"), rasterized_semantics, fps=10)
        elif opt.save == "mp4":
            imageio.mimwrite(os.path.join(output_dir, f"{seed:03d}_uncond.mp4"), frames, fps=20, quality=9)
            if not opt.stitch:
                imageio.mimwrite(os.path.join(output_dir, f"{seed:03d}_smpl.mp4"), rasterized_semantics, fps=20, quality=9)
        elif opt.save == "png":
            imageio.imwrite(os.path.join(output_dir, f"{seed:03d}_uncond.png"), np.concatenate(frames, axis=1))
            imageio.imwrite(os.path.join(output_dir, f"{seed:03d}_smpl.png"), np.concatenate(rasterized_semantics, axis=1))
        else:
            raise NotImplementedError