import os
import glob
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid
from lib.components.ema import ExponentialMovingAverage
from torch.utils.tensorboard import SummaryWriter

import configs
from lib import generators, discriminators
from lib.data import get_dataset, get_dataset_distributed, get_preprocessor
from lib.components import nv_misc
from lib.components import training_stats
from lib.data.augment import AugmentPipe


from tqdm import tqdm
import copy


def z_sampler(shape, device, dist):
    if dist == 'gaussian':
        z = torch.randn(shape, device=device)
    elif dist == 'uniform':
        z = torch.rand(shape, device=device) * 2 - 1
    return z


class BaseTrainer:

    def __init__(self, rank, world_size, device, opt, config):

        self.rank = rank
        self.world_size = world_size
        self.opt = opt

        self.device = device

        self.config = config

        self.output_dir = os.path.join(opt.output_dir, self.config["name"])
        os.makedirs(self.output_dir, exist_ok=True)

        self.meta = configs.extract_metadata(self.config, 0)

        self.amp = self.meta.get("use_mixed_precision", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)

        self.init_model(self.meta)
        print("rank {}: finished initializing model".format(rank))
        self.training_stats = training_stats.Collector(regex='.*')
        self.ada_stats = training_stats.Collector(regex='real_signs')
        self.init_optimizer(self.meta)

        if self.opt.set_step != None:
            self.generator.step = self.opt.set_step
            self.discriminator.step = self.opt.set_step

        self.generator.set_device(self.device)

        if rank == 0 or rank == world_size - 1:
            self.fixed_z = z_sampler((25, self.meta['latent_dim']), device='cpu', dist=self.meta['z_dist'])
            self.writer = SummaryWriter(log_dir=self.output_dir)

        self.skip_train = opt.skip_train


    def init_model(self, meta):

        if len(glob.glob(os.path.join(self.output_dir, "*.pth"))) > 0:
            generator_path = sorted(glob.glob(os.path.join(self.output_dir, '*generator.pth')))[-1]
            generator = torch.load(generator_path, map_location=self.device)
            discriminator_path = sorted(glob.glob(os.path.join(self.output_dir, '*discriminator.pth')))[-1]
            discriminator = torch.load(discriminator_path, map_location=self.device)
            ema_path = sorted(glob.glob(os.path.join(self.output_dir, '*ema.pth')))[-1]
            ema = torch.load(ema_path, map_location=self.device)
            augment_path = sorted(glob.glob(os.path.join(self.output_dir, '*augment.pth')))[-1]
            augment_pipe = torch.load(augment_path, map_location=self.device)

            if self.amp:
                scaler_path = sorted(glob.glob(os.path.join(self.output_dir, '*scaler.pth')))[-1]
                self.scaler.load_state_dict(torch.load(scaler_path, map_location=self.device))

            if self.rank == 0:
                print()
                print("loaded generator at step {}".format(generator.step))
                print("loaded discriminator at step {}".format(discriminator.step))
        else:
            generator = getattr(generators, meta['generator'])(**meta).to(self.device)
            print("rank {} finished building generator".format(self.rank))
            discriminator = getattr(discriminators, meta['discriminator'])(**meta).to(self.device)
            print("rank {} finished building discriminator".format(self.rank))
            ema = ExponentialMovingAverage(generator.parameters(), decay=0.999).to(self.device)
            print("rank {} finished building ema".format(self.rank))
            augment_pipe = AugmentPipe(**meta["ada_aug"]).to(self.device)
            print("rank {} finished building augment".format(self.rank))

        self.generator_ddp = DDP(generator, device_ids=[self.device], find_unused_parameters=True, broadcast_buffers=False)
        print("rank {} finished DDP(generator)".format(self.rank))
        self.discriminator_ddp = DDP(discriminator, device_ids=[self.device], find_unused_parameters=True, broadcast_buffers=False)
        print("rank {} finished DDP(discriminator)".format(self.rank))
        self.generator = self.generator_ddp.module
        self.discriminator = self.discriminator_ddp.module
        self.ema = ema
        self.ada_aug = augment_pipe
        # self.augment_pipe.requires_grad_(False)


    def init_optimizer(self, meta):

        if meta.get('unique_lr', False):
            mapping_network_parameters = [p for n, p in self.generator_ddp.named_parameters() if 'mapping_network' in n]
            generator_parameters = [p for n, p in self.generator_ddp.named_parameters() if 'mapping_network' not in n]
            self.optimizer_G = torch.optim.Adam([{'params': generator_parameters, 'name': 'generator'},
                                            {'params': mapping_network_parameters, 'name': 'mapping_network', 'lr': meta['gen_lr'] * meta['lr_multiplier']}],
                                           lr=meta['gen_lr'], betas=meta['betas'], weight_decay=meta['weight_decay'])
        else:
            self.optimizer_G = torch.optim.Adam(self.generator_ddp.parameters(), lr=meta['gen_lr'], betas=meta['betas'], weight_decay=meta['weight_decay'])

        self.optimizer_D = torch.optim.Adam(self.discriminator_ddp.parameters(), lr=meta['disc_lr'], betas=meta['betas'], weight_decay=meta['weight_decay'])

        if len(glob.glob(os.path.join(self.output_dir, "*.pth"))) > 0:
            optimizer_G_path = sorted(glob.glob(os.path.join(self.output_dir, '*optimizer_G.pth')))[-1]
            self.optimizer_G.load_state_dict(torch.load(optimizer_G_path))
            optimizer_D_path = sorted(glob.glob(os.path.join(self.output_dir, '*optimizer_D.pth')))[-1]
            self.optimizer_D.load_state_dict(torch.load(optimizer_D_path))


    def reset_augment(self, meta):

        self.augment_p_delta = meta['ada_interval'] * meta['batch_size'] / (meta['ada_kimg'] * 1000)
        self.ada_aug.p.copy_(torch.as_tensor(0.))


    def update_augment(self, meta):

        self.ada_stats.update()
        delta = np.sign(self.ada_stats['real_signs'] - meta['ada_target']) * self.augment_p_delta
        new_p = torch.max(self.ada_aug.p + delta, nv_misc.constant(0, device=self.device))
        new_p = torch.min(new_p, nv_misc.constant(1, device=self.device))
        self.ada_aug.p.copy_(new_p)


    def write_options(self):

        with open(os.path.join(self.output_dir, 'options.txt'), 'w') as f:
            f.write(str(self.opt))
            f.write('\n\n')
            f.write(str(self.generator))
            f.write('\n\n')
            f.write(str(self.discriminator))
            f.write('\n\n')
            f.write(str(self.config))


    def set_learning_rate(self, meta):

        for param_group in self.optimizer_G.param_groups:
            if param_group.get('name', None) == 'mapping_network':
                param_group['lr'] = meta['gen_lr'] * meta['lr_multiplier']
            else:
                param_group['lr'] = meta['gen_lr']
            param_group['betas'] = meta['betas']
            param_group['weight_decay'] = meta['weight_decay']
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = meta['disc_lr']
            param_group['betas'] = meta['betas']
            param_group['weight_decay'] = meta['weight_decay']


    def set_batch_size(self, meta):

        div = meta["batch_size"] // self.world_size
        self.proc_batch_size = div
        self.batch_split = meta["batch_split"] * self.opt.bs_factor
        self.batch_size = meta["batch_size"]


    def save_model(self):

        # delete old auto-save models
        saved_models = [_ for _ in os.listdir(self.output_dir) if _.endswith(".pth")]
        for model in saved_models:
            if int(model[:8]) % self.opt.model_keep_interval != 0:
                os.remove(os.path.join(self.output_dir, model))

        steps = self.discriminator.step
        torch.save(self.ema, os.path.join(self.output_dir, f'{steps:08d}_ema.pth'))
        torch.save(self.ada_aug, os.path.join(self.output_dir, f'{steps:08d}_augment.pth'))
        torch.save(self.generator_ddp.module, os.path.join(self.output_dir, f'{steps:08d}_generator.pth'))
        torch.save(self.discriminator_ddp.module, os.path.join(self.output_dir, f'{steps:08d}_discriminator.pth'))
        torch.save(self.optimizer_G.state_dict(), os.path.join(self.output_dir, f'{steps:08d}_optimizer_G.pth'))
        torch.save(self.optimizer_D.state_dict(), os.path.join(self.output_dir, f'{steps:08d}_optimizer_D.pth'))

        if self.amp:
            torch.save(self.scaler.state_dict(), os.path.join(self.output_dir, f'{steps:08d}_scaler.pth'))

        torch.cuda.empty_cache()

    def log_scalar(self, meta):

        self.writer.add_scalar("train/g_signs", self.training_stats['gen_signs'], self.discriminator.step)
        self.writer.add_scalar("train/r_signs", self.training_stats['real_signs'], self.discriminator.step)
        self.writer.add_scalar("train/ada_p", self.ada_aug.p.detach().item(), self.discriminator.step)


    def log_weights(self):

        inv_scale = 1. / self.scaler.get_scale()

        for tag, value in self.generator_ddp.named_parameters():
            self.writer.add_histogram("train/weights/gen/" + tag, value.cpu(), self.discriminator.step)
            if value.grad is not None:
                self.writer.add_histogram("train/grad/gen/" + tag, (value.grad * inv_scale).cpu(), self.discriminator.step)

        for tag, value in self.discriminator_ddp.named_parameters():
            self.writer.add_histogram("train/weights/disc/" + tag, value.cpu(), self.discriminator.step)
            if value.grad is not None:
                self.writer.add_histogram("train/grad/disc/" + tag, (value.grad * inv_scale).cpu(), self.discriminator.step)


    def log_image(self, alpha, meta):

        assert self.rank == 0

        self.generator_ddp.eval()

        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.no_grad():
                copied_metadata = copy.deepcopy(meta)
                copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                copied_metadata['gen_height'] = 2 * max(64, meta["gen_height"])
                copied_metadata['gen_width'] = copied_metadata["gen_height"] * meta["gen_width"] // meta["gen_height"]
                gen_imgs = \
                self.generator_ddp.module.staged_forward(self.fixed_z.to(self.device), **copied_metadata)[0]
        grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
        self.writer.add_image("train/fixed", grid, self.discriminator.step)

        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.no_grad():
                copied_metadata = copy.deepcopy(meta)
                copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                copied_metadata['h_mean'] += copied_metadata.get('vis_rotate', 0.5)
                copied_metadata['gen_height'] = 2 * max(64, meta["gen_height"])
                copied_metadata['gen_width'] = copied_metadata["gen_height"] * meta["gen_width"] // meta["gen_height"]
                gen_imgs = \
                self.generator_ddp.module.staged_forward(self.fixed_z.to(self.device), **copied_metadata)[0]
        grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
        self.writer.add_image("train/tilted", grid, self.discriminator.step)

        self.ema.store(self.generator_ddp.parameters())
        self.ema.copy_to(self.generator_ddp.parameters())

        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.no_grad():
                copied_metadata = copy.deepcopy(meta)
                copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                copied_metadata['gen_height'] = 2 * max(64, meta["gen_height"])
                copied_metadata['gen_width'] = copied_metadata["gen_height"] * meta["gen_width"] // meta["gen_height"]
                gen_imgs = \
                self.generator_ddp.module.staged_forward(self.fixed_z.to(self.device), **copied_metadata)[0]
        grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
        self.writer.add_image("train/fixed_ema", grid, self.discriminator.step)

        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.no_grad():
                copied_metadata = copy.deepcopy(meta)
                copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                copied_metadata['h_mean'] += copied_metadata.get('vis_rotate', 0.5)
                copied_metadata['gen_height'] = 2 * max(64, meta["gen_height"])
                copied_metadata['gen_width'] = copied_metadata["gen_height"] * meta["gen_width"] // meta["gen_height"]
                gen_imgs = \
                self.generator_ddp.module.staged_forward(self.fixed_z.to(self.device), **copied_metadata)[0]
        grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
        self.writer.add_image("train/tilted_ema", grid, self.discriminator.step)

        random_z = torch.randn_like(self.fixed_z).to(self.device)
        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.no_grad():
                copied_metadata = copy.deepcopy(meta)
                copied_metadata['gen_height'] = 2 * max(64, meta["gen_height"])
                copied_metadata['gen_width'] = copied_metadata["gen_height"] * meta["gen_width"] // meta["gen_height"]
                copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                copied_metadata['truncation_psi'] = 0.7
                gen_imgs, depths = self.generator_ddp.module.staged_forward(random_z, **copied_metadata)
        grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
        self.writer.add_image("train/random", grid, self.discriminator.step)
        grid = make_grid(depths[:25], nrow=5, normalize=True)
        self.writer.add_image("train/random_depth", grid, self.discriminator.step)

        with torch.cuda.amp.autocast(enabled=self.amp):
            with torch.no_grad():
                copied_metadata = copy.deepcopy(meta)
                copied_metadata['h_stddev'] = copied_metadata['v_stddev'] = 0
                copied_metadata['h_mean'] += copied_metadata.get('vis_rotate', 0.5)
                copied_metadata['gen_height'] = 2 * max(64, meta["gen_height"])
                copied_metadata['gen_width'] = copied_metadata["gen_height"] * meta["gen_width"] // meta["gen_height"]
                copied_metadata['truncation_psi'] = 0.7
                gen_imgs, depths = \
                self.generator_ddp.module.staged_forward(random_z, **copied_metadata)
        grid = make_grid(gen_imgs[:25], nrow=5, normalize=True)
        self.writer.add_image("train/random_tilted", grid, self.discriminator.step)
        grid = make_grid(depths[:25], nrow=5, normalize=True)
        self.writer.add_image("train/random_tilted_depth", grid, self.discriminator.step)

        self.ema.restore(self.generator_ddp.parameters())


    def log_data(self, data, meta):

        train_vis = make_grid(data["images"][:9], 3, normalize=True)
        self.writer.add_image("train/data", train_vis, self.discriminator.step)

    def train_discriminator(self, data, alpha, meta):

        raise NotImplementedError


    def train_generator(self, data, alpha, meta):

        raise NotImplementedError


    def run(self):

        self.write_options()

        torch.manual_seed(self.rank)
        dataloader = None
        total_progress_bar = tqdm(total=self.opt.n_epochs, desc="Total progress", dynamic_ncols=True)
        total_progress_bar.update(self.discriminator.epoch)
        interior_step_bar = tqdm(dynamic_ncols=True)

        for _ in range(self.opt.n_epochs):

            total_progress_bar.update(1)
            meta = configs.extract_metadata(self.config, self.discriminator.step)
            if "batch_size" not in meta: break

            if not dataloader or any([self.batch_size != meta["batch_size"],
                                      self.gen_height != meta["gen_height"],
                                      self.gen_width != meta["gen_width"]]):

                self.set_batch_size(meta)
                self.gen_height = meta["gen_height"]
                self.gen_width = meta["gen_width"]

                tqdm.write("global/local {}/{}: getting dataloader".format(self.rank, self.rank))
                dataloader, CHANNELS = get_dataset_distributed(meta['dataset'], self.world_size, self.rank, self.proc_batch_size, **meta)

                step_next_upsample = configs.next_upsample_step(self.config, self.discriminator.step)
                step_last_upsample = configs.last_upsample_step(self.config, self.discriminator.step)

                interior_step_bar.reset(total=(step_next_upsample - step_last_upsample))
                interior_step_bar.set_description(f"Progress to next stage")
                interior_step_bar.update((self.discriminator.step - step_last_upsample))

            self.set_learning_rate(meta)
            self.reset_augment(meta)

            tqdm.write("global/local {}/{}: start new epoch".format(self.rank, self.rank))
            for i, data in enumerate(dataloader):

                meta = configs.extract_metadata(self.config, self.discriminator.step)
                if "batch_size" not in meta: break
                if self.batch_size != meta["batch_size"]: break
                if self.gen_height != meta["gen_height"]: break
                if self.gen_width != meta["gen_width"]: break

                if self.amp and self.scaler.get_scale() < 1:
                    self.scaler.update(1.)

                if  self.rank == 0 and self.discriminator.step % 1000 == 0 and self.discriminator.step > 0:
                    self.save_model()

                self.generator_ddp.train()
                self.discriminator_ddp.train()

                alpha = min(1, (self.discriminator.step - step_last_upsample) / meta['fade_steps'])
                meta['nerf_noise'] = max(0, 1. - self.discriminator.step / 5000.)

                if not self.skip_train:

                    ###########################
                    ### TRAIN DISCRIMINATOR ###
                    ###########################
                    d_loss = self.train_discriminator(data, alpha, meta)

                    #######################
                    ### TRAIN GENERATOR ###
                    #######################

                    g_loss, topk_num = self.train_generator(data, alpha, meta)

                else:

                    d_loss = 0.
                    g_loss, topk_num = 0, 0

                # UPDATE ADA
                if not meta['ada_interval'] == 0 and alpha >= meta['ada_alpha_thresh'] and i % meta['ada_interval'] == 0:
                    self.update_augment(meta)

                # UPDATE PROGRESS, SAVE RESULTS

                if self.rank == 0:

                    interior_step_bar.update(1)

                    if i % 10 == 0:

                        self.training_stats.update()
                        tqdm.write(f"[Exp: {self.output_dir}]"
                                   f"[Epoch: {self.discriminator.epoch}/{self.opt.n_epochs}]"
                                   f"[D loss: {d_loss:.3f}]"
                                   f"[G loss: {g_loss:.3f}]"
                                   f"[Step: {self.discriminator.step}]"
                                   f"[Alpha: {alpha:.2f}]"
                                   f"[Res: ({meta['gen_width']}, {meta['gen_height']})]"
                                   f"[Batch Size: {meta['batch_size']}]"
                                   f"[TopK: {topk_num}]"
                                   f"[G_Signs: {self.training_stats['gen_signs']:.3f}]"
                                   f"[R_Signs: {self.training_stats['real_signs']:.3f}]"
                                   f"[ADA_P: {self.ada_aug.p.detach().item():.3f}]"
                                   f"[Scale: {self.scaler.get_scale()}]")

                        self.writer.add_scalar("train/g_loss", g_loss, self.discriminator.step)
                        self.writer.add_scalar("train/d_loss", d_loss, self.discriminator.step)
                        self.writer.add_scalar("train/alpha", alpha, self.discriminator.step)
                        self.writer.add_scalar("train/topk_num", topk_num, self.discriminator.step)
                        self.writer.add_scalar("train/scaler", self.scaler.get_scale(), self.discriminator.step)

                        self.log_scalar(meta)

                    if self.discriminator.step % self.opt.sample_interval == 0 and self.discriminator.step > 0:

                        self.log_image(alpha, meta)
                        self.log_data(data, meta)
                        self.log_weights()

                self.discriminator.step += 1
                self.generator.step += 1

            self.discriminator.epoch += 1
            self.generator.epoch += 1