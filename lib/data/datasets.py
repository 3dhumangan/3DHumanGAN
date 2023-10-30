"""Datasets"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import json
import pickle
from scipy.spatial.transform import Rotation
from PIL import Image
import math
import joblib
import cv2


def apply_transformation(points, transform):
    points_homo = np.pad(points, [[0, 0], [0, 1]], "constant", constant_values=1.)
    points = np.einsum("ij,bj->bi", transform, points_homo)[:, :3]
    return points


class SHHQDataset(Dataset):

    corrupted = [118464]

    def __init__(self, **kwargs):
        super().__init__()

        self.coodinate_mode = kwargs.get("coordinate_mode", "fix_body")

        self.root = kwargs["dataroot"]
        self.length = kwargs["dataset_length"]
        self.height = kwargs["gen_height"]
        self.width = kwargs["gen_width"]
        self.joints = kwargs.get("joints", [])
        self.latent_dim = kwargs["latent_dim"]

        self.inference = kwargs.get("inference", False)
        self.image_only = kwargs.get("image_only", False)
        self.geo_only = kwargs.get("geo_only", False)
        self.condition_only = kwargs.get("condition_only", False)

        self.vertex_downsample = kwargs.get("vertex_downsample", 1)

        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize((self.height, self.width), interpolation=Image.BILINEAR)])

        self.semantics_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.height, self.width), interpolation=Image.NEAREST)])

        self.smpl_tpose_vertices, self.smpl_faces = self.get_smpl()

    def __len__(self):
        return self.length

    def get_all_latents(self):

        if os.path.exists("./datasets/all_inversions.pkl"):
            print("using cached inversions")
            latents = joblib.load("./datasets/all_inversions.pkl")
            latents = latents[:self.length]
            latents = latents[:, :self.latent_dim]
            latents = 2 * latents
        elif self.use_ceph:
            latents = self.ceph.get_pkl(os.path.join(self.root, "all_inversions.pkl"))
            joblib.dump(latents, "./datasets/all_inversions.pkl")
            latents = latents[:self.length]
            latents = latents[:, :self.latent_dim]
            latents = 2 * latents
        else:
            pbar = tqdm.tqdm(total=len(self))
            pbar.set_description("loading initial latents")
            latents = np.zeros([len(self), self.latent_dim], dtype=np.float32)
            for i in range(len(self)):
                # match distribution to normal(mean=0, std=1)
                latent_path = os.path.join(self.root, "inversions", f"{i + 1:06d}.npy")
                latent = np.load(latent_path)
                latents[i] = 2 * latent[:self.latent_dim]
                pbar.update(1)

        return latents


    def get_smpl(self):

        print("SHHQDataset: loading SMPL")

        smpl_path = os.path.join("./datasets", "SMPL_NEUTRAL.pkl")

        with open(smpl_path, "rb") as f:
            smpl = pickle.load(f, encoding='latin1')

        smpl_tpose_vertices = smpl["v_template"]
        smpl_faces = smpl["f"]

        return smpl_tpose_vertices, smpl_faces


    def preprocess_smpl(self, pred):

        if self.coodinate_mode == "fix_camera":
            return self._preprocess_smpl_fix_camera(pred)
        elif self.coodinate_mode == "fix_body":
            return self._preprocess_smpl_fix_body(pred)
        else:
            raise NotImplementedError


    def _preprocess_smpl_fix_body(self, pred):

        fov = np.pi * 12 / 180
        focal = 1. / np.tan(fov / 2)

        sx, sy, tx, ty = pred['orig_cam'][0].astype(np.float32)
        sx = sx / 2.
        skeleton_xyz = pred['joints'][0].astype(np.float32)
        skeleton_xyz = skeleton_xyz[self.joints]

        K = np.array([
            [focal, 0, 0, 0],
            [0, focal, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        R = np.eye(4)
        T = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, focal / sx],
            [0, 0, 0, 1],
        ])

        body_pose = pred["full_pose"][0]
        tpose_vertices_shaped = pred['tpose_vertices'][0]
        fk_matrices = pred['fk_matrices'][0]

        inverse_root = np.linalg.inv(body_pose[0])
        cano_rotation = Rotation.from_euler("xyz", [np.pi, 0, 0]).as_matrix()

        cano_matrix = np.eye(4)
        cano_matrix[:3, :3] = cano_rotation @ inverse_root
        fk_matrices = np.einsum("ij,bjk->bik", cano_matrix, fk_matrices)

        lbs_weights = pred['lbs_weights']
        vertice_fk_matrices = np.einsum("bi,ijk->bjk", lbs_weights, fk_matrices)
        tpose_vertices_homo = np.pad(tpose_vertices_shaped, [[0 ,0], [0, 1]], "constant", constant_values=1.)
        vertices = np.einsum("bij,bj->bi", vertice_fk_matrices, tpose_vertices_homo)[:, :3]

        skeleton_xyz = apply_transformation(skeleton_xyz, cano_matrix)

        tpose_vertices = self.smpl_tpose_vertices.astype(np.float32)
        tpose_vertices[..., 1] += 0.35

        output = {
            "scales": sx,
            "skeletons_xyz": skeleton_xyz.astype(np.float32),
            "intrinsics": K.astype(np.float32),
            "vertices": vertices.astype(np.float32),
            "tpose_vertices": tpose_vertices,
            "full_pose": pred["full_pose"][0].astype(np.float32),
            "fk_matrices": fk_matrices.astype(np.float32),
            "lbs_weights": lbs_weights.astype(np.float32),
            "cano_matrices": cano_matrix.astype(np.float32),
            "R": R.astype(np.float32),
            "T": T.astype(np.float32),
        }

        if self.inference:
            output.update({
                "body_shape": pred["betas"][0].astype(np.float32)
            })

        return output

    def _preprocess_smpl_fix_camera(self, pred):

        fov = np.pi * 12 / 180
        focal = 1. / np.tan(fov / 2)

        sx, sy, tx, ty = pred['orig_cam'][0].astype(np.float32)
        sx = sx / 2.
        skeleton_xyz = pred['joints'][0].astype(np.float32)
        skeleton_xyz = skeleton_xyz[self.joints]

        K = np.array([
            [focal, 0, 0, 0],
            [0, focal, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        R = np.eye(4)
        T = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, focal / sx],
            [0, 0, 0, 1],
        ])

        tpose_vertices_shaped = pred['tpose_vertices'][0]
        fk_matrices = pred['fk_matrices'][0]

        world2cam_matrix = R @ T
        cam2world_matrix = np.linalg.inv(world2cam_matrix)

        tpose_vertices = self.smpl_tpose_vertices.astype(np.float32)
        tpose_vertices[..., 1] += 0.35

        output = {
            "scales": sx,
            "skeletons_xyz": skeleton_xyz.astype(np.float32),
            "intrinsics": K.astype(np.float32),
            "tpose_vertices": tpose_vertices,
            "tpose_vertices_shaped": tpose_vertices_shaped.astype(np.float32),
            "full_pose": pred["full_pose"][0].astype(np.float32),
            "fk_matrices": fk_matrices.astype(np.float32),
            "lbs_weights": pred['lbs_weights'].astype(np.float32),
            "cam2world_matrices": cam2world_matrix.astype(np.float32),
            "R": R.astype(np.float32),
            "T": T.astype(np.float32),
        }

        if self.inference:
            output.update({
                "body_shape": pred["betas"][0].astype(np.float32)
            })

        return output


    def _get_item_image_only(self, index):

        while index in self.corrupted:
            index = (index + 1) % len(self)

            rgb_path = os.path.join(self.root, "images", f"{index + 1:06d}.png")
            mask_path = os.path.join(self.root, "masks", f"{index + 1:06d}.png")
            rgb = np.array(Image.open(rgb_path))
            mask = np.array(Image.open(mask_path))

        data = {}

        if self.geo_only:
            mask = np.stack([mask, mask, mask], axis=-1)
            mask = self.image_transform(mask)
            data.update({"images": mask, "masks": mask, })
        else:
            rgb[mask == 0] = 255
            rgb = self.image_transform(rgb)
            mask = self.image_transform(mask)
            data.update({"images": rgb, "masks": mask, })

        return data


    def _get_item_condition_only(self, index):

        smpl_path = os.path.join(self.root, "smpl", f"{index + 1:06d}.pkl")

        smpl = joblib.load(smpl_path)

        smpl_output = self.preprocess_smpl(smpl)

        return smpl_output


    def __getitem__(self, index):

        while index in self.corrupted:
            index = (index + 1) % len(self)

        if self.image_only: return self._get_item_image_only(index)
        if self.condition_only: return self._get_item_condition_only(index)

        rgb_path = os.path.join(self.root, "images", f"{index + 1:06d}.png")
        mask_path = os.path.join(self.root, "masks", f"{index + 1:06d}.png")
        latent_path = os.path.join(self.root, "inversions", f"{index + 1:06d}.npy")
        body_seg_path = os.path.join(self.root, "body_seg", f"{index + 1:06d}.png")
        rgb = np.array(Image.open(rgb_path))
        mask = np.array(Image.open(mask_path))
        body_segments = np.array(Image.open(body_seg_path))
        latent = 2 * np.load(latent_path)[:self.latent_dim]

        data = { "indices": index, "latents": latent.astype(np.float32) }

        if self.geo_only:
            mask = np.stack([mask, mask, mask], axis=-1)
            mask = self.image_transform(mask)
            data.update({ "images": mask, "masks": mask, })
        else:
            rgb[mask==0] = 255
            rgb = self.image_transform(rgb)
            mask = self.image_transform(mask)
            data.update({ "images": rgb, "masks": mask, })

        # body segments

        body_segments = cv2.resize(body_segments[:, :, 0], (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        seg_fg = (body_segments > 0)
        body_segments[seg_fg] += 1  # 0 is reserved for "fake"
        body_segments[~seg_fg] = 1  # 1 is reserved for "background"
        data.update({ "body_segments": body_segments.astype(np.int64) })

        # smpl

        if len(self.joints) > 0:

            smpl_path = os.path.join(self.root, "smpl", f"{index + 1:06d}.pkl")
            smpl = joblib.load(smpl_path)
            smpl_output = self.preprocess_smpl(smpl)
            data.update(smpl_output)

        return data
