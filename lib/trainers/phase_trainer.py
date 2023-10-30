import os
import glob
import math
import numpy as np
import random

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid

from lib.components import training_stats
import copy

from lib.trainers.base_trainer import BaseTrainer, z_sampler
from lib.components.perceptual_loss import VGGPerceptualLoss
from lib.components.util import normalize_2nd_moment
from lib.data import get_dataset, get_dataset_distributed, get_preprocessor


class PhaseTrainer (BaseTrainer):

    def __init__(self, rank, world_size, device, opt, config):

        super().__init__(rank, world_size, device, opt, config)

        self.set_batch_size(self.meta)
        self.init_condition_sampler(self.meta)

        if len(glob.glob(os.path.join(self.output_dir, "*.pth"))) == 0:
            print("rank {} initializing latent codes".format(rank))
            appearance_codes = torch.from_numpy(self.condition_loader.dataset.get_all_latents())
            self.generator_ddp.module.latent_pool.init(appearance_codes)

        if self.meta["gen_height"] // self.meta["gen_width"] == 2:
            self.first_layer_size = (4, 2)
        elif self.meta["gen_width"] // self.meta["gen_height"] == 2:
            self.first_layer_size = (2, 4)
        else:
            self.first_layer_size = (4, 4)

        self.init_loss_modules(self.meta)

        self.preprocessor = get_preprocessor(self.condition_loader, self.meta).to(self.device)
        self.preprocessor.device = self.device

        torch.cuda.empty_cache()

        print("rank {} finished trainer init".format(rank))


    def init_loss_modules(self, meta):

        if sum(meta["perceptual_lambda"]) > 0:
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)


    def init_optimizer(self, meta):

        named_params = list(self.generator_ddp.named_parameters())
        neural_field_mapping_params = { n: p for n, p in named_params if 'neural_field_mapping_network' in n}
        synthesis_mapping_params = { n: p for n, p in named_params if 'synthesis_mapping_network' in n}
        apperance_codes = { n: p for n, p in named_params if 'latent_pool' in n}
        neural_field_params = { n: p for n, p in named_params if 'neural_field' in n and n not in neural_field_mapping_params}
        generator_params = { n: p for n, p in named_params if n not in {**apperance_codes, **neural_field_params, **neural_field_mapping_params, **synthesis_mapping_params}}

        self.optimizer_G = torch.optim.Adam([
                {'params': list(generator_params.values()), 'name': 'generator'},
                {'params': list(apperance_codes.values()), 'name': 'appearance_codes',
                 'lr': meta['gen_lr'] * meta['appearance_codes_lr_mul']},
                {'params': list(neural_field_mapping_params.values()), 'name': 'neural_field_mapping',
                 'lr': meta['gen_lr'] * meta['mapping_net_lr_mul']},
                {'params': list(synthesis_mapping_params.values()), 'name': 'synthesis_mapping',
                 'lr': meta['gen_lr']},
                {'params': list(neural_field_params.values()), 'name': 'neural_field',
                 'lr': meta['gen_lr'] * meta['neural_field_lr_mul']}],
            lr=meta['gen_lr'], betas=meta['betas'], weight_decay=meta['weight_decay'])

        self.optimizer_D = torch.optim.Adam(self.discriminator_ddp.parameters(), lr=meta['disc_lr'], betas=meta['betas'], weight_decay=meta['weight_decay'])

        if len(glob.glob(os.path.join(self.output_dir, "*.pth"))) > 0:
            optimizer_G_path = sorted(glob.glob(os.path.join(self.output_dir, '*optimizer_G.pth')))[-1]
            self.optimizer_G.load_state_dict(torch.load(optimizer_G_path, map_location=self.device))
            optimizer_D_path = sorted(glob.glob(os.path.join(self.output_dir, '*optimizer_D.pth')))[-1]
            self.optimizer_D.load_state_dict(torch.load(optimizer_D_path, map_location=self.device))


    def set_learning_rate(self, meta):

        for param_group in self.optimizer_G.param_groups:
            if param_group['name'] == 'neural_field_mapping':
                param_group['lr'] = meta['gen_lr'] * meta['mapping_net_lr_mul']
            elif param_group['name'] == 'synthesis_mapping':
                param_group['lr'] = meta['gen_lr']
            elif param_group['name'] == 'neural_field':
                param_group['lr'] = meta['gen_lr'] * meta['neural_field_lr_mul']
            elif param_group['name'] == 'appearance_codes':
                param_group['lr'] = meta['gen_lr'] * meta['appearance_codes_lr_mul']
            elif param_group['name'] == 'generator':
                param_group['lr'] = meta['gen_lr']
            else:
                raise ValueError("param_group not defined")

            param_group['betas'] = meta['betas']
            param_group['weight_decay'] = meta['weight_decay']

        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = meta['disc_lr']
            param_group['betas'] = meta['betas']
            param_group['weight_decay'] = meta['weight_decay']


    def log_scalar(self, meta):

        self.writer.add_scalar("train/ada_p", self.ada_aug.p.detach().item(), self.discriminator.step)
        for name in self.training_stats.names():
            self.writer.add_scalar(f"train/{name}", self.training_stats[name], self.discriminator.step)


    def init_condition_sampler(self, meta):

        print("rank {} initializing condition sampler".format(self.rank))

        dataloader, _ = get_dataset_distributed(
            meta['dataset'], self.world_size, self.rank, self.proc_batch_size, shuffle=True, as_condition_sampler=True, **meta)
        self.condition_loader = dataloader
        self.condition_sampler = iter(dataloader)


    def sample_conditions(self, num_samples, meta, device="cpu"):

        num_sampled = 0
        images = []
        conditions = {}

        while num_sampled < num_samples:

            try:
                data = next(self.condition_sampler)
            except:
                self.init_condition_sampler(meta)
                data = next(self.condition_sampler)

            images.append(data["images"])
            for k, v in data.items():
                if k == "images": continue
                if k in conditions:
                    conditions[k].append(v)
                else:
                    conditions[k] = [v]

            num_sampled += len(data["images"])

        images = torch.cat(images, dim=0)[:num_samples].to(device, non_blocking=True)
        for k, v in conditions.items():
            conditions[k] = torch.cat(v, dim=0)[:num_samples].to(device, non_blocking=True)

        conditions["images"] = images

        return images, conditions


    def _get_disc_input_real(self, real_images, alpha, phase, meta):

        if meta.get("dual_discrimination", False):
            with torch.no_grad():
                alpha_disc = alpha
                rescale_height = meta["render_height"] * math.pow(2, alpha_disc * math.log2(
                    meta["gen_height"] / meta["render_height"]))
                rescale_height = int(round(rescale_height))
                rescale_width = meta["gen_width"] * rescale_height // meta["gen_height"]
                real_images_render = F.interpolate(real_images, (meta["render_height"], meta["render_width"]),
                                                   mode="bilinear")
                real_images_render = F.interpolate(real_images_render, (meta["gen_height"], meta["gen_width"]),
                                                   mode="bilinear")
                real_images_synthesis = F.interpolate(real_images, (rescale_height, rescale_width), mode="bilinear")
                real_images_synthesis = F.interpolate(real_images_synthesis, (meta["gen_height"], meta["gen_width"]),
                                                      mode="bilinear")
                disc_input_real = torch.cat([real_images_render, real_images_synthesis], dim=1)
                training_stats.report("rescale_height", rescale_height)
        elif "render" in phase["gen_modal"]:
            with torch.no_grad():
                real_images = F.interpolate(real_images, (meta["render_height"], meta["render_width"]), mode="bilinear")
                # real_images = F.interpolate(real_images, (meta["gen_height"], meta["gen_width"]), mode="bilinear")
                disc_input_real = real_images
        else:
            disc_input_real = real_images

        return disc_input_real


    def _get_disc_input_gen(self, gen_outputs, alpha, phase, meta):

        if meta.get("dual_discrimination", False):
            gen_images_render = F.interpolate(gen_outputs["rgbs_render"], (meta["gen_height"], meta["gen_width"]), mode="bilinear")
            gen_images_rgbs = gen_outputs["rgbs"]
            disc_input_gen = torch.cat([gen_images_render, gen_images_rgbs], dim=1)
        else:
            disc_input_gen = gen_outputs[phase["gen_modal"]]

        return disc_input_gen


    def _calculate_segmentation_loss(self, segments, gt_segments, meta):
        """
        segments: batch_size, num_labels, height, width
        gt_segments: batch_size, height, width
        """

        mode = meta.get("segmentation_loss_mode", "cross_entropy_balanced")
        prior_weights = meta.get("segmentation_weights", [1. for _ in range(meta["label_dim"])])
        prior_weights = torch.tensor(prior_weights, dtype=segments.dtype, device=self.device)
        prior_weights = prior_weights / prior_weights.mean()

        B, _, H, W = segments.shape

        if gt_segments.shape[1] != H or gt_segments.shape[2] != W:
            with torch.no_grad():
                gt_segments = gt_segments.unsqueeze(1).float()
                gt_segments = F.interpolate(gt_segments, (H, W), mode="nearest")
                gt_segments = gt_segments.squeeze(1).long()

        if mode == "cross_entropy":
            loss = F.cross_entropy(segments, gt_segments)
        elif mode == "cross_entropy_multiclass":
            gt_one_hot = F.one_hot(gt_segments, num_classes=meta['label_dim']).permute(0, 3, 1, 2)
            gt_one_hot[:, 1, :, :][gt_segments > 0] = 1
            loss = F.binary_cross_entropy_with_logits(segments, gt_one_hot.to(segments.dtype))
        elif mode == "cross_entropy_balanced":
            if torch.any(gt_segments > 0):
                gt_one_hot = F.one_hot(gt_segments, num_classes=meta['label_dim']).permute(0, 3, 1, 2)
                class_occurence = torch.sum(gt_one_hot, dim=(0, 2, 3))
                class_occurence[0] = 0
                num_classes_occur = torch.count_nonzero(class_occurence)
                coefficients = torch.reciprocal(class_occurence.to(segments.dtype)) * torch.numel(gt_one_hot) / (num_classes_occur * gt_one_hot.shape[1])
                coefficients[0] = 0
                coefficients[torch.isinf(coefficients)] = 0
                coefficients = coefficients * prior_weights
                weight_map = coefficients[gt_segments]
                loss = F.cross_entropy(segments, gt_segments, reduction="none")
                loss = (loss * weight_map).mean()
            else:
                loss = F.cross_entropy(segments, gt_segments)
        elif mode == "softplus":
            gt_one_hot = F.one_hot(gt_segments, num_classes=meta['label_dim']).permute(0, 3, 1, 2)
            gt_one_hot[:, 1, :, :][gt_segments > 0] = 1
            match = (gt_one_hot > 0)
            loss = segments.clone()
            loss[match] = -segments[match]
            loss = F.softplus(loss[:, 0]).mean() + F.softplus(loss[:, 1]).mean() + F.softplus(loss[:, 2:]).mean()
            loss = loss / 3

        real_prob = (1 - torch.softmax(segments, dim=1)[:, 0, :, :]).mean()
        pred_labels = torch.argmax(segments[:, 1:, :, :], dim=1) + 1
        accurracy = (pred_labels == gt_segments).float().mean()

        return loss, accurracy, real_prob


    def _calculate_r1_regularization(self, disc_input_real, disc_output_real, phase, meta):

        if meta["gan_lambda"] > 0:

            grad_real = torch.autograd.grad(
                outputs=self.scaler.scale(disc_output_real["prediction"].sum()),
                inputs=disc_input_real,
                create_graph=True)[0]

        elif meta["segmentation_lambda"] > 0:

            pred_segments_real = disc_output_real["segments"]

            pred_segments_real = torch.softmax(pred_segments_real, dim=1)
            grad_real = torch.autograd.grad(
                outputs=self.scaler.scale(pred_segments_real.sum()),
                inputs=disc_input_real,
                create_graph=True)[0]

        else:
            raise Exception("Cannot do r1 regularization when segmentation_lambda == 0 and gan_lambda == 0")

        inv_scale = 1. / self.scaler.get_scale()
        grad_real = [p * inv_scale for p in grad_real][0]

        # print_stats(grad_real.unsqueeze(-1), "grad_real", scientific=True)

        with torch.cuda.amp.autocast(enabled=self.amp):
            grad_penalty = grad_real.view(grad_real.size(0), -1).pow(2).sum(dim=1)
            grad_penalty = grad_penalty.mean()
            grad_penalty = 0.5 * meta['r1_lambda'] * grad_penalty

        if torch.any(torch.isnan(grad_penalty)):
            grad_penalty = 0.

        return grad_penalty


    def train_discriminator(self, data, alpha, meta):

        phase_idx = self.discriminator.step % len(meta['phases'])
        phase = meta['phases'][phase_idx]

        self.optimizer_D.zero_grad()

        data = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}
        data = self.preprocessor(data, phase["rotate"], **meta)

        d_loss = self._train_discriminator(data, alpha, meta, phase)

        # print("D {} loss = {}".format(phase, d_loss))

        if isinstance(d_loss, torch.Tensor):

            self.scaler.scale(d_loss).backward()
            self.scaler.unscale_(self.optimizer_D)
            torch.nn.utils.clip_grad_norm_(self.discriminator_ddp.parameters(), meta['grad_clip'])
            self.scaler.step(self.optimizer_D)

        return d_loss.detach().item()


    def train_generator(self, data, alpha, meta):

        phase_idx = self.discriminator.step % len(meta['phases'])
        phase = meta['phases'][phase_idx]

        self.optimizer_G.zero_grad()

        data = {k: v.to(self.device, non_blocking=True) for k, v in data.items()}
        data = self.preprocessor(data, phase["rotate"], **meta)

        g_loss, topk_num = self._train_generator(data, alpha, meta, phase)

        # print("G {} loss = {}".format(phase, g_loss))

        self.scaler.unscale_(self.optimizer_G)
        torch.nn.utils.clip_grad_norm_(self.generator_ddp.parameters(), meta['grad_clip'])
        self.scaler.step(self.optimizer_G)
        self.scaler.update()
        self.ema.update(self.generator_ddp.parameters())

        return g_loss, topk_num


    def _train_discriminator(self, data, alpha, meta, phase):

        gan_lambda = meta["gan_lambda"]
        segmentation_lambda = meta["segmentation_lambda"]

        real_images = data["images"]
        real_segments = data["body_segments"]

        if phase["rotate"] or random.random() < 0.5:
            real_segments = data["rasterized_segments"]

        with torch.cuda.amp.autocast(enabled=self.amp):
            # Generate images for discriminator training

            with torch.no_grad():

                # if phase["uncond"]:
                z = z_sampler((real_images.shape[0], meta['latent_dim']), device=self.device, dist=meta['z_dist'])

                split_batch_size = z.shape[0] // self.batch_split
                gen_outputs = {}

                for split in range(self.batch_split):

                    subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                    subset_conditions = {k: v[split * split_batch_size:(split + 1) * split_batch_size]
                                         for k, v in data.items()}
                    g_output = self.generator_ddp(subset_z, subset_conditions,
                                                  disable_synthesis=(phase["gen_modal"] != "rgbs"),
                                                  latent_indices=(None if phase["uncond"] else subset_conditions["indices"]), **meta)

                    for k, v in g_output.items():
                        if k in gen_outputs:
                            gen_outputs[k].append(v)
                        else:
                            gen_outputs[k] = [v]

                for k in gen_outputs.keys():
                    gen_outputs[k] = torch.cat(gen_outputs[k], dim=0)

            disc_input_real = self._get_disc_input_real(real_images, alpha, phase, meta)
            disc_input_real.requires_grad = True

            disc_mode = "real" if phase["rotate"] else ("real" if random.random() > meta.get("disc_mode_switch_p", 0.5) else "gen")
            disc_output_real = self.discriminator_ddp(disc_input_real, data, alpha=alpha, mode=disc_mode, **meta)
            pred_real = disc_output_real["prediction"]
            if gan_lambda > 0: training_stats.report("real_signs_" + phase["name"], pred_real.sign())

        grad_penalty = 4 * self._calculate_r1_regularization(disc_input_real, disc_output_real, phase, meta) if phase["do_r1"] else 0.
        if phase["do_r1"] and self.rank == 0:
            training_stats.report('r1_' + phase["name"], grad_penalty / 4.)

        with torch.cuda.amp.autocast(enabled=self.amp):

            disc_mode = "gen" if phase["rotate"] else ("gen" if random.random() > meta.get("disc_mode_switch_p", 0.5) else "real")

            disc_input_gen = self._get_disc_input_gen(gen_outputs, alpha, phase, meta)
            disc_output_gen = self.discriminator_ddp(disc_input_gen, data, alpha=alpha, mode=disc_mode, **meta)

            pred_gen = disc_output_gen["prediction"]
            if gan_lambda > 0:
                gan_loss = gan_lambda * (torch.nn.functional.softplus(pred_gen).mean() + torch.nn.functional.softplus(-pred_real).mean())
            else:
                gan_loss = pred_gen.sum() * 0 + pred_real.sum() * 0

            if "segments" in disc_output_real:
                if meta["segmentation_lambda"] > 0:
                    segmentation_loss_real, acc_real, prob_real = self._calculate_segmentation_loss(disc_output_real["segments"], real_segments, meta)
                    segmentation_loss_gen, _, prob_gen = self._calculate_segmentation_loss(disc_output_gen["segments"], torch.zeros_like(real_segments), meta)
                    segmentation_loss = (segmentation_loss_real + segmentation_loss_gen) * segmentation_lambda
                    training_stats.report("d_segmentation_loss", segmentation_loss)
                    training_stats.report("segmentation_acc_real", acc_real)
                    training_stats.report("segmentation_prob_real", prob_real)
                    training_stats.report("segmentation_prob_gen", prob_gen)
                    training_stats.report("segmentation_signs", math.log((prob_real.item() + 1e-3) / (prob_gen.item() + 1e-3)))
                else:
                    segmentation_loss = (disc_output_real["segments"].sum() + disc_output_gen["segments"].sum()) * 0
            else:
                segmentation_loss = 0

            if "latents" in disc_output_gen:
                if meta["latent_lambda"] > 0:
                    with torch.no_grad():
                        gt_latents_real = normalize_2nd_moment(data["latents"])
                        if phase["uncond"]:
                            gt_latents_gen = normalize_2nd_moment(z)
                        else:
                            gt_latents_gen = normalize_2nd_moment(self.generator_ddp.module.latent_pool(data["indices"]))
                    latent_loss = F.smooth_l1_loss(normalize_2nd_moment(disc_output_gen["latents"]), gt_latents_gen, beta=0.1) + \
                                  F.smooth_l1_loss(normalize_2nd_moment(disc_output_real["latents"]), gt_latents_real, beta=0.1)
                    latent_loss = latent_loss * meta["latent_lambda"]
                    training_stats.report("d_latent_loss", latent_loss)
                else:
                    latent_loss = (disc_output_real["latents"].sum() + disc_output_gen["latents"].sum()) * 0
            else:
                latent_loss = 0

            d_loss = gan_loss + grad_penalty + segmentation_loss + latent_loss

        return d_loss


    def _train_generator(self, data, alpha, meta, phase, loss_scale=1.):

        gan_lambda = meta["gan_lambda"] if phase["uncond"] else 0
        segmentation_lambda = meta["segmentation_lambda"]

        real_images = data["images"]

        z = z_sampler((real_images.shape[0], meta['latent_dim']), device=self.device, dist=meta['z_dist'])

        real_z = data["latents"]
        split_batch_size = real_images.shape[0] // self.batch_split

        g_loss_total = 0.

        for split in range(self.batch_split):

            with torch.cuda.amp.autocast(enabled=self.amp):

                subset_z = z[split * split_batch_size:(split + 1) * split_batch_size]
                subset_real_images = real_images[split * split_batch_size:(split + 1) * split_batch_size]
                subset_real_z = real_z[split * split_batch_size:(split + 1) * split_batch_size]
                subset_conditions = {k: v[split * split_batch_size:(split + 1) * split_batch_size]
                                     for k, v in data.items()}
                gen_outputs = self.generator_ddp(subset_z, subset_conditions,
                                                 disable_synthesis=(phase["gen_modal"] != "rgbs"),
                                                 latent_indices=(None if phase["uncond"] else subset_conditions["indices"]), **meta)

                disc_mode = "gen" if phase["rotate"] else ("gen" if random.random() > meta.get("disc_mode_switch_p", 0.5) else "real")

                disc_input_gen = self._get_disc_input_gen(gen_outputs, alpha, phase, meta)

                disc_output = self.discriminator_ddp(disc_input_gen, subset_conditions, alpha=alpha, mode=disc_mode, **meta)
                pred_gen = disc_output["prediction"]

                if gan_lambda > 0: training_stats.report("gen_signs_" + phase["name"], pred_gen.sign())

                if 'topk_interval' in meta and 'topk_v' in meta:
                    topk_percentage = max(0.99 ** (self.discriminator.step / meta['topk_interval']), meta['topk_v'])
                else:
                    topk_percentage = 1.

                topk_num = math.ceil(topk_percentage * pred_gen.shape[0])

                pred_gen = torch.topk(pred_gen, topk_num, dim=0).values

                if gan_lambda > 0:
                    gan_loss = gan_lambda * torch.nn.functional.softplus(-pred_gen).mean()
                else:
                    gan_loss = 0 * pred_gen.sum()

                if meta['latent_lambda'] > 0:

                    with torch.no_grad():
                        if phase["uncond"]:
                            gt_latents_gen = normalize_2nd_moment(subset_z)
                        else:
                            gt_latents_gen = self.generator_ddp.module.latent_pool(subset_conditions["indices"])
                            gt_latents_gen = normalize_2nd_moment(gt_latents_gen)
                    latent_loss = F.smooth_l1_loss(normalize_2nd_moment(disc_output["latents"]), gt_latents_gen, beta=0.1)

                    if not phase["uncond"]:
                        latent_loss += F.smooth_l1_loss(subset_z, subset_real_z.detach(), beta=0.1)

                    latent_loss = latent_loss * meta['latent_lambda']
                    training_stats.report("g_latent_loss", latent_loss)

                else:
                    latent_loss = disc_output["latents"].sum() * 0 if "latents" in disc_output else 0.

                if not phase["uncond"] and sum(meta['perceptual_lambda']) > 0:
                    perceptual_input_gen = 0.5 * gen_outputs[phase["gen_modal"]] + 0.5
                    perceptual_input_real = 0.5 * subset_real_images + 0.5
                    perceptual_losses = self.perceptual_loss(perceptual_input_gen, perceptual_input_real.detach())
                    perceptual_losses = [meta['perceptual_lambda'][i] * perceptual_losses[i] for i in range(4)]
                    perceptual_loss = sum(perceptual_losses)
                    training_stats.report("perceptual_loss", perceptual_loss)
                else:
                    perceptual_loss = 0

                if not phase["uncond"] and meta['photometric_lambda'] > 0:
                    photometric_loss = F.smooth_l1_loss(gen_outputs[phase["gen_modal"]], subset_real_images.detach(), beta=0.1) * meta['photometric_lambda']
                    training_stats.report("photometric_loss", photometric_loss)
                else:
                    photometric_loss = 0

                if "segments" in disc_output:
                    if meta["segmentation_lambda"] > 0:
                        gt_modal = "rasterized_segments" if phase["rotate"] else ("rasterized_segments" if random.random() < 0.5 else "body_segments")
                        real_segments = subset_conditions[gt_modal]
                        segmentation_loss, _, _ = self._calculate_segmentation_loss(disc_output["segments"], real_segments, meta)
                        segmentation_loss = segmentation_loss * segmentation_lambda
                        training_stats.report("g_segmentation_loss", segmentation_loss)
                    else:
                        segmentation_loss = disc_output["segments"].sum() * 0
                else:
                    segmentation_loss = 0

                g_loss = gan_loss + perceptual_loss + photometric_loss + latent_loss + segmentation_loss
                g_loss = g_loss * loss_scale / self.batch_split

                g_loss_scaled = self.scaler.scale(g_loss)
                g_loss_scaled.backward(retain_graph=True)
                g_loss_total += g_loss.detach().item()

        return g_loss_total, topk_num


    @torch.no_grad()
    def log_image(self, alpha, meta):

        def visualize_segmentation(segments):

            if len(segments.shape) == 4:
                labels = torch.argmax(segments, dim=1, keepdim=True)
            else:
                labels = segments.unsqueeze(1)

            B, C, H, W = labels.shape

            labels_float = labels.float() / (meta["label_dim"] - 1)
            labels_float = labels_float * 2 - 1
            labels_float = 2 * labels_float
            blue = torch.clamp(-labels_float, 0, 1)
            red = torch.clamp(labels_float, 0, 1)
            green = torch.clamp(2-torch.abs(labels_float), 0, 1)
            segmentation_mask = torch.cat([red, green, blue], dim=1)

            fake_mask = (labels == 0).expand(B, 3, H, W)
            segmentation_mask[fake_mask] = -0.5
            bg_mask = (labels == 1).expand(B, 3, H, W)
            segmentation_mask[bg_mask] = 1

            return segmentation_mask

        n_rows = 2
        n_per_row = 4
        n_total = n_rows * n_per_row

        assert self.rank == 0

        self.generator_ddp.eval()
        self.discriminator_ddp.eval()

        real_images, conditions = self.sample_conditions(n_total, meta, self.device)
        conditions = self.preprocessor(conditions, False, **meta)

        # cond_z = self.generator_ddp.module.latent_pool(conditions["indices"])
        random_z = z_sampler((real_images.shape[0], meta['latent_dim']), device=self.device, dist=meta['z_dist'])

        self.ema.store(self.generator_ddp.parameters())
        self.ema.copy_to(self.generator_ddp.parameters())

        # copied_meta = copy.deepcopy(meta)
        # copied_meta['gen_height'] = 2 * max(64, meta["gen_height"])
        # copied_meta['gen_width'] = copied_meta["gen_height"] * meta["gen_width"] // meta["gen_height"]
        # copied_meta['h_stddev'] = copied_meta['v_stddev'] = 0
        # copied_meta['truncation_psi'] = 0.7
        # phase = copied_meta['phases'][0]
        # with torch.cuda.amp.autocast(enabled=self.amp):
        #     gen_outputs = self.generator_ddp.module.staged_forward(
        #         cond_z, conditions, **copied_meta)
        #     if meta["segmentation_lambda"] > 0:
        #         disc_input_real = self._get_disc_input_real(real_images, alpha, phase, meta)
        #         disc_outputs_real = self.discriminator_ddp(disc_input_real, conditions, alpha=1.)
        #         disc_input_gen = self._get_disc_input_gen(gen_outputs, alpha, phase, meta)
        #         disc_outputs_gen = self.discriminator_ddp(disc_input_gen, conditions, alpha=1.)
        #
        # grid = make_grid(0.5 * torch.clamp(gen_outputs["rgbs"], -1., 1.) + 0.5, nrow=n_per_row, normalize=False)
        # self.writer.add_image("train/cond_synthesis", grid, self.discriminator.step)
        # grid = make_grid(0.5 * torch.clamp(gen_outputs["rgbs_render"], -1., 1.) + 0.5, nrow=n_per_row, normalize=False)
        # self.writer.add_image("train/cond_render", grid, self.discriminator.step)
        # grid = make_grid(gen_outputs["depths"], nrow=n_per_row, normalize=True)
        # self.writer.add_image("train/cond_depth", grid, self.discriminator.step)
        # # grid = make_grid(gen_outputs["heatmaps"], nrow=n_per_row, normalize=True)
        # # self.writer.add_image("train/heatmap", grid, self.discriminator.step)
        # self.writer.add_image("train/condition_mesh", grid, self.discriminator.step)

        # if meta["segmentation_lambda"] > 0:
        #     grid = make_grid(visualize_segmentation(conditions[meta["condition_modal_disc_gen"]]), nrow=n_per_row, normalize=False)
        #     self.writer.add_image("train/condition", grid, self.discriminator.step)
        #     grid = make_grid(visualize_segmentation(disc_outputs_real["segments"]), nrow=n_per_row, normalize=False)
        #     self.writer.add_image("train/real_segments", grid, self.discriminator.step)
        #     grid = make_grid(visualize_segmentation(disc_outputs_gen["segments"]), nrow=n_per_row, normalize=False)
        #     self.writer.add_image("train/cond_segments", grid, self.discriminator.step)

        copied_meta = copy.deepcopy(meta)
        copied_meta['gen_height'] = 2 * max(64, meta["gen_height"])
        copied_meta['gen_width'] = copied_meta["gen_height"] * meta["gen_width"] // meta["gen_height"]
        copied_meta['h_stddev'] = copied_meta['v_stddev'] = 0
        copied_meta['truncation_psi'] = 0.7
        phase = copied_meta['phases'][0]
        with torch.cuda.amp.autocast(enabled=self.amp):
            gen_outputs = self.generator_ddp.module.staged_forward(
                random_z, conditions, **copied_meta)
            if meta["segmentation_lambda"] > 0:
                disc_input_gen = self._get_disc_input_gen(gen_outputs, alpha, phase, meta)
                disc_outputs_gen = self.discriminator_ddp(disc_input_gen, conditions, alpha=1.)
        grid = make_grid(0.5 * torch.clamp(gen_outputs["rgbs"], -1., 1.) + 0.5, nrow=n_per_row, normalize=False)
        self.writer.add_image("train/uncond_synthesis", grid, self.discriminator.step)
        grid = make_grid(0.5 * torch.clamp(gen_outputs["rgbs_render"], -1., 1.) + 0.5, nrow=n_per_row, normalize=False)
        self.writer.add_image("train/uncond_render", grid, self.discriminator.step)
        grid = make_grid(gen_outputs["depths"], nrow=n_per_row, normalize=True)
        self.writer.add_image("train/uncond_depth", grid, self.discriminator.step)
        if meta["segmentation_lambda"] > 0:
            grid = make_grid(visualize_segmentation(disc_outputs_gen["segments"]), nrow=n_per_row, normalize=False)
            self.writer.add_image("train/uncond_segments", grid, self.discriminator.step)


        copied_meta = copy.deepcopy(meta)
        copied_meta['gen_height'] = 2 * max(64, meta["gen_height"])
        copied_meta['gen_width'] = copied_meta["gen_height"] * meta["gen_width"] // meta["gen_height"]
        copied_meta['h_stddev'] = copied_meta['v_stddev'] = 0
        copied_meta['h_mean'] = math.pi / 6
        copied_meta['v_mean'] = 0
        copied_meta['truncation_psi'] = 0.7
        phase = copied_meta['phases'][0]
        conditions = self.preprocessor(conditions, True, **copied_meta)
        with torch.cuda.amp.autocast(enabled=self.amp):
            gen_outputs = self.generator_ddp.module.staged_forward(
                random_z, conditions, **copied_meta)
            if meta["segmentation_lambda"] > 0:
                disc_input_gen = self._get_disc_input_gen(gen_outputs, alpha, phase, meta)
                disc_outputs_gen = self.discriminator_ddp(disc_input_gen, conditions, alpha=1.)
        grid = make_grid(0.5 * torch.clamp(gen_outputs["rgbs"], -1., 1.) + 0.5, nrow=n_per_row, normalize=False)
        self.writer.add_image("train/uncond_rotate_synthesis", grid, self.discriminator.step)
        grid = make_grid(0.5 * torch.clamp(gen_outputs["rgbs_render"], -1., 1.) + 0.5, nrow=n_per_row, normalize=False)
        self.writer.add_image("train/uncond_rotate_render", grid, self.discriminator.step)
        grid = make_grid(gen_outputs["depths"], nrow=n_per_row, normalize=True)
        self.writer.add_image("train/uncond_rotate_depth", grid, self.discriminator.step)

        if meta["segmentation_lambda"] > 0:
            grid = make_grid(visualize_segmentation(conditions[meta["condition_modal_disc_gen"]]), nrow=n_per_row, normalize=False)
            self.writer.add_image("train/condition_rotate", grid, self.discriminator.step)
            grid = make_grid(visualize_segmentation(disc_outputs_gen["segments"]), nrow=n_per_row, normalize=False)
            self.writer.add_image("train/uncond_rotate_segments", grid, self.discriminator.step)

        self.ema.restore(self.generator_ddp.parameters())
        torch.cuda.empty_cache()
