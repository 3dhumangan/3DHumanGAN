import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

from lib.components.util import LatentPool
from lib.components.smpl import get_geo_features
import lib.generators.volume_rendering as vr
import lib.components.map3d_layers as layers
from lib.components.mapping_networks import MappingNetwork, TwoPartMappingNetwork


class SynthesisNetwork(nn.Module):

    def __repr__(self):
        return f"{self.__class__.__name__}({self.repr})"

    def __init__(self, input_dim, style_dim, hidden_dim=256, num_blocks=8, mod_blocks=list(range(8)), name_prefix='m3d', spatial_normalization="instance_norm", map3d_mode="isolated", **kwargs):

        super().__init__()

        self.repr = f"input_dim={input_dim}, style_dim={style_dim}, hidden_dim={hidden_dim}, " \

        self.style_dim = style_dim
        self.num_blocks = num_blocks
        self.mod_blocks = mod_blocks
        self.map3d_mode = map3d_mode

        self.normalization = spatial_normalization

        _out_dim = input_dim

        network = OrderedDict()
        to_rgbs = OrderedDict()

        for i in range(num_blocks):

            _block_name = f'{name_prefix}_{i}'

            _in_dim = _out_dim
            _out_dim = hidden_dim

            if spatial_normalization == "none":
                _block = layers.SynthesisBlock(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, name_prefix=_block_name)
            else:
                _block = layers.SPADEBlock(in_dim=_in_dim, out_dim=_out_dim, style_dim=style_dim, normalization=spatial_normalization)

            network[_block_name] = _block

            _to_rgb = layers.ToRGB(in_dim=_out_dim, dim_rgb=3, use_conv=(spatial_normalization != "none"))
            to_rgbs[_block_name] = _to_rgb

        self.network = nn.ModuleDict(network)
        self.to_rgbs = nn.ModuleDict(to_rgbs)


    def forward(self, input, style, fixed_style, rgb=None, return_internal=False, **kwargs):

        x = input

        if hasattr(self, "normalization") and self.normalization == "none":
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).view(B, H*W, C)

        output = {}

        for idx, (name, block) in enumerate(self.network.items()):

            if self.map3d_mode == "all":
                B, _, C = fixed_style.shape
                input_style = style + fixed_style.view(B, C, 1, 1).expand_as(style)
            elif self.map3d_mode == "mixed":
                B, _, C = fixed_style.shape
                inject_style = style if idx in self.mod_blocks else torch.zeros_like(style)
                input_style = inject_style + fixed_style.view(B, C, 1, 1).expand_as(inject_style)
            elif self.map3d_mode == "isolated":
                input_style = style if idx in self.mod_blocks else fixed_style
            else:
                raise ValueError("invalid map3d_mode")

            x = block(x, input_style, skip=(idx >= self.num_blocks // 2))
            # print_stats(x.unsqueeze(-1), f"x after layer {name}")

            if idx >= self.num_blocks // 2 - 1:
                rgb = self.to_rgbs[name](x, rgb)

            if return_internal:
                output.update({name + "_feature_map": x,
                               name + "_rgb": rgb})

        if hasattr(self, "normalization") and self.normalization == "none":
            rgb = rgb.permute(0, 2, 1).view(B, 3, H, W)

        output.update({"final": rgb})

        return output



class Map3DGenerator(nn.Module):

    def __init__(self, neural_field_cls, **kwargs):
        super().__init__()

        self.latent_dim = kwargs['latent_dim']
        self.hidden_dim = kwargs['hidden_dim']
        self.feature_dim = kwargs['feature_dim']
        self.geo_feature_dim = kwargs['geo_feature_dim']
        self.label_dim = kwargs['label_dim']
        self.gen_height = kwargs['gen_height']
        self.gen_width = kwargs['gen_width']
        self.disable_modulation = kwargs.get('disable_modulation', False)
        self.legacy_mode = kwargs.get('legacy_mode', False)

        self.neural_field = neural_field_cls(
            output_dim=kwargs['feature_dim'] + 4,
            latent_dim=kwargs['latent_dim'],
            input_dim=kwargs['input_dim'],
            hidden_dim=kwargs['hidden_dim'],
            geo_feature_dim=kwargs['geo_feature_dim'],
            feature_dim=kwargs['feature_dim'],
            num_blocks=kwargs['neural_field_blocks'],
            device=None
        )

        self.synthesis_input = layers.SynthesisInput(
            input_dim=2 + (kwargs["semantic_dim"] if kwargs.get('2d_semantic_input', False) else 0) +
                      (1 if kwargs.get('2d_label_input', False) else 0),
            output_dim=kwargs['feature_dim'],
        )

        self.synthesis_style_input = layers.SynthesisStyleInput(
            input_dim=1 if 'segments' in kwargs['condition_modal_gen'] else 3,
            latent_dim=kwargs['latent_dim'],
            output_dim=kwargs['feature_dim'],
            num_layers=3
        )

        self.synthesis_network = SynthesisNetwork(
            input_dim=kwargs['feature_dim'] + (kwargs['latent_dim'] if kwargs.get('2d_latent_input', False) else 0),
            style_dim=kwargs['feature_dim'],
            hidden_dim=kwargs['hidden_dim'],
            num_blocks=kwargs['synthesis_blocks'],
            mod_blocks=kwargs['mod_blocks'],
            map3d_mode=kwargs.get('map3d_mode', 'isolated'),
            spatial_normalization=kwargs.get('spatial_normalization', 'instance_norm')
        )

        self.neural_field_mapping_network = MappingNetwork(
            latent_dim=kwargs['latent_dim'],
            map_hidden_dim=kwargs['hidden_dim'],
            map_output_dim=2*kwargs['neural_field_blocks']*kwargs['hidden_dim']
        )

        self.synthesis_mapping_network = TwoPartMappingNetwork(
            z_dim=kwargs['latent_dim'],
            c_dim=0,
            implicit_dim=1,
            w_dim=kwargs['feature_dim'],
            num_ws=1,
            trunk_layers=7,
            branch_layers=1,
            lr_multiplier=0.01
        )

        self.epoch = 0
        self.step = 0

        self.side_length = kwargs["side_length"]
        self.geo_feature_dim = kwargs['geo_feature_dim']

        self.latent_pool = LatentPool(kwargs["dataset_length"], kwargs["latent_dim"])


    def set_device(self, device):

        self.device = device
        self.neural_field.device = device


    def generate_avg_latent(self):
        """Calculates average freq and phase shifts"""

        z = torch.randn((10000, self.latent_dim), device=self.neural_field.device)
        neural_field_freq, neural_field_phase = self.neural_field_mapping_network(z)
        _, synthesis_styles = self.synthesis_mapping_network(z)
        self.avg_latent = (
            z.mean(dim=0, keepdim=True),
            neural_field_freq.mean(dim=0, keepdim=True),
            neural_field_phase.mean(dim=0, keepdim=True),
            synthesis_styles.mean(dim=0, keepdim=True))

        return self.avg_latent

    @torch.no_grad()
    def get_geo_features(self, points, skeletons, vertices, tpose_vertices, fk_matrices, lbs_weights):

        if self.disable_modulation:
            modulation = torch.zeros(list(points.shape)[:2] + [self.geo_feature_dim], dtype=points.dtype,
                                     device=self.device)
        else:
            modulation = get_geo_features(points, skeletons, vertices, tpose_vertices, fk_matrices, lbs_weights, self.legacy_mode)

        return modulation


    def forward(self, latent, conditions, render_height, render_width, latent_indices=None, **kwargs):

        num_steps = kwargs.get('num_steps', 24)

        batch_size = latent.shape[0]

        if latent_indices is not None:
            latent = self.latent_pool(latent_indices)

        if kwargs.get("neural_field_latent_input", True):
            neural_field_freq, neural_field_phase = self.neural_field_mapping_network(latent)
        else:
            neural_field_freq, neural_field_phase = self.neural_field_mapping_network(torch.zeros_like(latent))

        _, synthesis_styles = self.synthesis_mapping_network(latent)

        if kwargs.get("disable_render", False):

            condition_modal = kwargs["condition_modal_gen"]
            condition = conditions[condition_modal]
            if "segments" in condition_modal:
                condition = condition.unsqueeze(1).to(latent.dtype) / (kwargs["label_dim"] - 1) * 2 - 1

            if kwargs.get("spade_latent_input", True):
                feature_maps = self.synthesis_style_input(condition, latent)
            else:
                feature_maps = self.synthesis_style_input(condition, torch.zeros_like(latent))

            rgb_render = torch.zeros([batch_size, 3, render_height, render_width], dtype=latent.dtype, device=self.device)

        else:

            rgb_render, feature_maps, depths, weights, extrinsics = self.render(
                neural_field_freq, neural_field_phase, conditions, render_width, render_height,
                coarse_steps=num_steps, fine_steps=num_steps, **kwargs)

        feature_map_interpolation = kwargs.get("feature_map_interpolation", "bilinear")
        feature_maps = F.interpolate(feature_maps, (self.gen_height, self.gen_width), mode=feature_map_interpolation)

        if kwargs.get("disable_synthesis", False):

            output = {
                "rgbs": rgb_render,
                "rgbs_render": rgb_render,
            }

        else:

            coords_input = self.synthesis_input.get_2d_coords(batch_size, self.gen_height, self.gen_width, dtype=latent.dtype, device=self.device)

            if kwargs.get('2d_label_input', False):
                coords_input = torch.cat([coords_input, conditions["rasterized_segments"].unsqueeze(1) / self.label_dim * 2 - 1], dim=1)

            synthesis_input = self.synthesis_input(coords_input)

            if kwargs.get('2d_latent_input', False):
                latent_exp = latent.view(batch_size, self.latent_dim, 1, 1).expand(batch_size, self.latent_dim, self.gen_height, self.gen_width)
                synthesis_input = torch.cat([synthesis_input, latent_exp], dim=1)

            synthesis_output = self.synthesis_network(synthesis_input, feature_maps, synthesis_styles, **kwargs)

            # print_stats(neural_field_freq.unsqueeze(-1), "neural_field_freq")
            # print_stats(synthesis_styles.unsqueeze(-1), "synthesis_styles")
            # print_stats(feature_maps.unsqueeze(-1), "feature_maps")
            # print_stats(rgb_render.unsqueeze(-1), "rgb_render")
            # print_stats(rgb_synthesis.unsqueeze(-1), "rgb_synthesis")

            output = {
                "rgbs": synthesis_output["final"],
                "rgbs_render": rgb_render,
            }

        return output

    def staged_forward(self, latent, conditions, render_height, render_width, truncation_psi, **kwargs):

        num_steps = kwargs.get('num_steps', 24)

        batch_size = latent.shape[0]

        if kwargs.get("neural_field_latent_input", True):
            neural_field_freq, neural_field_phase = self.neural_field_mapping_network(latent)
        else:
            neural_field_freq, neural_field_phase = self.neural_field_mapping_network(torch.zeros_like(latent))

        _, synthesis_styles = self.synthesis_mapping_network(latent)

        if truncation_psi < 1.:
            self.generate_avg_latent()
            avg_latent, avg_neural_field_freq, avg_neural_field_phase, avg_synthesis_styles = self.avg_latent
            neural_field_freq = avg_neural_field_freq + truncation_psi * (neural_field_freq - avg_neural_field_freq)
            neural_field_phase = avg_neural_field_phase + truncation_psi * (neural_field_phase - avg_neural_field_phase)
            latent = avg_latent + truncation_psi * (latent - avg_latent)
            synthesis_styles = avg_synthesis_styles + truncation_psi * (synthesis_styles - avg_synthesis_styles)

        if kwargs.get("disable_render", False):

            condition_modal = kwargs["condition_modal_gen"]
            condition = conditions[condition_modal]
            if "segments" in condition_modal:
                condition = condition.unsqueeze(1).to(latent.dtype) / (kwargs["label_dim"] - 1) * 2 - 1

            if kwargs.get("spade_latent_input", True):
                feature_maps = self.synthesis_style_input(condition, latent)
            else:
                feature_maps = self.synthesis_style_input(condition, torch.zeros_like(latent))

            rgb_render = torch.zeros([batch_size, 3, render_height, render_width], dtype=latent.dtype, device=self.device)
            depths = torch.zeros([batch_size, render_height * render_width, 1], dtype=latent.dtype, device=self.device)

        else:

            rgb_render, feature_maps, depths, weights, extrinsics = self.render(
                neural_field_freq, neural_field_phase, conditions, render_width, render_height,
                coarse_steps=num_steps, fine_steps=num_steps, staged=True, **kwargs)

        feature_map_interpolation = kwargs.get("feature_map_interpolation", "bilinear")
        feature_maps = F.interpolate(feature_maps, (self.gen_height, self.gen_width), mode=feature_map_interpolation)

        if kwargs.get("disable_synthesis", False):

            rgb_synthesis = F.interpolate(rgb_render, (self.gen_height, self.gen_width), mode='bilinear')
            output = {
                "rgbs": rgb_synthesis,
                "rgbs_render": rgb_render,
            }

        else:

            coords_input = self.synthesis_input.get_2d_coords(batch_size, self.gen_height, self.gen_width,
                                                              dtype=latent.dtype, device=self.device)

            # coords_input[:, 0, :, :] -= 0.5
            # feature_maps = F.pad(feature_maps, (0, 0, self.gen_height // 4, 0), mode="constant", value=0)
            # feature_maps = feature_maps[:, :, :self.gen_height, :]

            if kwargs.get('2d_label_input', False):
                coords_input = torch.cat([coords_input, conditions["rasterized_segments"].unsqueeze(1) / self.label_dim * 2 - 1], dim=1)

            synthesis_input = self.synthesis_input(coords_input)

            if kwargs.get('2d_latent_input', False):
                latent_exp = latent.view(batch_size, self.latent_dim, 1, 1).expand(batch_size, self.latent_dim,
                                                                                   self.gen_height, self.gen_width)
                synthesis_input = torch.cat([synthesis_input, latent_exp], dim=1)

            synthesis_output = self.synthesis_network(synthesis_input, feature_maps, synthesis_styles, **kwargs)

            output = {
                "rgbs": synthesis_output["final"],
                "rgbs_render": rgb_render,
            }

            if len(synthesis_output) > 1:
                output.update(synthesis_output)

        with torch.no_grad():
            focals = conditions["intrinsics"][:, 0, 0]
            scales = conditions["scales"].float()
            z_centers = focals / scales
            depth = depths - z_centers.view(batch_size, 1, 1)
            depth = depth / (kwargs["depth_length"] / 2.)
            depth = torch.clamp(depth, -1., 1.)
            depth_map = depth.reshape(batch_size, render_height, render_width).unsqueeze(1).contiguous().cpu()

        output.update({
            "depths": depth_map,
            "skeletons": conditions["skeletons_xyz"],
        })

        return output


    def render(self, freq, phase, conditions, render_width, render_height, ray_start, ray_end, coarse_steps, fine_steps, h_stddev, v_stddev, h_mean, v_mean, hierarchical_sample, sample_dist=None, lock_view_dependence=False, staged=False, max_points=50000, **kwargs):
        """
        Generates images from a noise vector, rendering parameters, and camera distribution.
        Uses the hierarchical sampling scheme described in NeRF.
        skeleton: [B, N_joints, 3]
        """

        skeletons = conditions["skeletons_xyz"]
        vertices = conditions["vertices"]
        tpose_vertices = conditions["tpose_vertices"]
        fk_matrices = conditions["fk_matrices"]
        lbs_weights = conditions["lbs_weights"]
        cam2world_matrices = conditions["cam2world_matrices"]
        focals = conditions["intrinsics"][:, 0, 0]
        scales = conditions["scales"].float()

        batch_size = freq.shape[0]

        # Generate initial camera rays and sample points.
        with torch.no_grad():

            points_cam, z_vals, rays_d_cam = vr.get_initial_rays_weak_perspective(
                focals, scales, coarse_steps,
                resolution=(render_width, render_height), device=self.device,
                ray_start=ray_start, ray_end=ray_end)
            # batch_size, pixels, num_steps, 1

            transformed_points, z_vals, transformed_ray_directions, transformed_ray_origins, pitch, yaw, extrinsics = \
                vr.transform_sampled_points(points_cam, z_vals, rays_d_cam,
                                            cam2world_matrix=cam2world_matrices,
                                            device=self.device, mode=sample_dist)
            transformed_points = transformed_points.reshape(batch_size, render_width * render_height * coarse_steps, 3)

            # print()
            # print_stats(transformed_points, "transformed_points")

            transformed_ray_directions_exp = vr.expand_ray_directions(transformed_ray_directions, coarse_steps)
            if lock_view_dependence:
                transformed_ray_directions_exp = torch.zeros_like(transformed_ray_directions_exp)
                transformed_ray_directions_exp[..., -1] = -1

        query_points = transformed_points

        if not staged:

            modulation = self.get_geo_features(query_points, skeletons, vertices, tpose_vertices, fk_matrices, lbs_weights)
            coarse_output = self.neural_field.forward(query_points, freq, phase, modulation,
                                                      ray_directions=transformed_ray_directions_exp,
                                                      input_scaler=2./self.side_length, modulation_scaler=1.)

        else:

            coarse_output = torch.zeros((batch_size, query_points.shape[1], self.feature_dim + 4), device=self.device)
            for b in range(batch_size):
                head = 0
                while head < transformed_points.shape[1]:
                    tail = head + max_points
                    modulation = self.get_geo_features(
                        query_points[b:b + 1, head:tail], skeletons[b:b + 1], vertices[b:b + 1],
                        tpose_vertices[b:b + 1], fk_matrices[b:b + 1], lbs_weights[b:b + 1])
                    coarse_output[b:b + 1, head:tail] = self.neural_field(
                        query_points[b:b + 1, head:tail], freq[b:b + 1], phase[b:b+1], modulation,
                        transformed_ray_directions_exp[b:b + 1, head:tail],
                        input_scaler=2. / self.side_length, modulation_scaler=1.)
                    head += max_points

        coarse_output = coarse_output.reshape(batch_size, render_width * render_height, coarse_steps, self.feature_dim + 4)

        # Re-sample fine points along camera rays, as described in NeRF
        if hierarchical_sample:
            with torch.no_grad():

                _, _, weights = vr.ray_integration(coarse_output, z_vals, device=self.device, clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

                weights = weights.reshape(batch_size * render_width * render_height, coarse_steps) + 1e-5

                #### Start new importance sampling
                z_vals = z_vals.reshape(batch_size * render_width * render_height, coarse_steps)
                z_vals_mid = 0.5 * (z_vals[: ,:-1] + z_vals[: ,1:])
                z_vals = z_vals.reshape(batch_size, render_width * render_height, coarse_steps, 1)
                fine_z_vals = vr.sample_pdf(z_vals_mid, weights[:, 1:-1], fine_steps, det=False).detach()
                fine_z_vals = fine_z_vals.reshape(batch_size, render_width * render_height, fine_steps, 1)

                fine_points = transformed_ray_origins.unsqueeze(2).contiguous() + transformed_ray_directions.unsqueeze(
                    2).contiguous() * fine_z_vals.expand(-1, -1, -1, 3).contiguous()
                fine_points = fine_points.reshape(batch_size, render_width * render_height * fine_steps, 3)

                if fine_steps < coarse_steps:
                    transformed_ray_directions_exp = transformed_ray_directions_exp.reshape(batch_size, render_width * render_height, coarse_steps, 3)
                    transformed_ray_directions_exp = transformed_ray_directions_exp[:, :, :fine_steps, :]
                    transformed_ray_directions_exp = transformed_ray_directions_exp.reshape(batch_size, render_width * render_height * fine_steps, 3)

                if lock_view_dependence:
                    transformed_ray_directions_exp = torch.zeros_like(transformed_ray_directions_exp)
                    transformed_ray_directions_exp[..., -1] = -1
                #### end new importance sampling

            # Model prediction on re-sampled find points
            if not staged:

                modulation = self.get_geo_features(fine_points, skeletons, vertices, tpose_vertices, fk_matrices, lbs_weights)
                fine_output = self.neural_field.forward(fine_points, freq, phase, modulation, ray_directions=transformed_ray_directions_exp,
                                                        input_scaler=2./self.side_length, modulation_scaler=1.)

            else:

                fine_output = torch.zeros((batch_size, fine_points.shape[1], self.feature_dim + 4), device=self.device)
                for b in range(batch_size):
                    head = 0
                    while head < fine_points.shape[1]:
                        tail = head + max_points
                        modulation = self.get_geo_features(
                            fine_points[b:b + 1, head:tail], skeletons[b:b + 1], vertices[b:b + 1],
                            tpose_vertices[b:b + 1], fk_matrices[b:b + 1], lbs_weights[b:b + 1])
                        fine_output[b:b + 1, head:tail] = self.neural_field(
                            fine_points[b:b + 1, head:tail], freq[b:b + 1], phase[b:b + 1], modulation,
                            ray_directions=transformed_ray_directions_exp[b:b + 1, head:tail],
                            input_scaler=2. / self.side_length, modulation_scaler=1.)
                        head += max_points

            fine_output = fine_output.reshape(batch_size, render_width * render_height, -1, self.feature_dim + 4)

            # Combine course and fine points
            all_outputs = torch.cat([fine_output, coarse_output], dim=-2)
            all_z_vals = torch.cat([fine_z_vals, z_vals], dim=-2)
            _, indices = torch.sort(all_z_vals, dim=-2)
            all_z_vals = torch.gather(all_z_vals, -2, indices)
            all_outputs = torch.gather(all_outputs, -2, indices.expand(-1, -1, -1, self.feature_dim + 4))
        else:
            all_outputs = coarse_output
            all_z_vals = z_vals


        # Perform ray integration
        render_outputs, depths, weights = vr.ray_integration(all_outputs, all_z_vals, device=self.device, white_back=kwargs.get('white_back', False), last_back=kwargs.get('last_back', False), clamp_mode=kwargs['clamp_mode'], noise_std=kwargs['nerf_noise'])

        render_outputs = render_outputs.reshape(batch_size, render_height, render_width, self.feature_dim + 3)
        render_outputs = render_outputs.permute(0, 3, 1, 2)

        rgb_render = render_outputs[:, :3] * 2 - 1.
        feature_maps = render_outputs[:, 3:]

        return rgb_render, feature_maps, depths, weights, extrinsics
