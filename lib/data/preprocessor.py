import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from lib.components.smpl import SMPL
import math

from pytorch3d.renderer import PerspectiveCameras, MeshRasterizer, RasterizationSettings
from pytorch3d.structures import Meshes
from pytorch3d.transforms import euler_angles_to_matrix


class SHHQPreprocessor(nn.Module):

    def __init__(self, gen_height, gen_width, **kwargs):
        super().__init__()

        self.height = gen_height
        self.width = gen_width
        self.device = "cpu"

        # self.x = nn.Linear(1, 1)
        self.mode = kwargs.get("coordinate_mode", "fix_body")

        self.register_buffer("vertex_approximation", torch.zeros([6890], dtype=torch.long))
        self.register_buffer("smpl_faces", torch.zeros([13776, 3], dtype=torch.long))
        self.register_buffer("smpl_faces_to_labels", torch.zeros([13776], dtype=torch.long))

        raster_settings = RasterizationSettings(
            image_size=(gen_height, gen_width),
            blur_radius=0.0, faces_per_pixel=1)

        self.rasterizer = MeshRasterizer(raster_settings=raster_settings)


    @torch.no_grad()
    def init_smpl(self, smpl_faces, smpl_faces_to_labels):

        self.smpl_faces.copy_(smpl_faces)
        self.smpl_faces_to_labels.copy_(smpl_faces_to_labels)


    @torch.no_grad()
    def forward(self, data, rotate=False, **kwargs):

        batch_size = data["scales"].shape[0]

        h_rotation = torch.randn(batch_size) * (kwargs["h_stddev"] if rotate else 0) + kwargs["h_mean"]
        v_rotation = torch.randn(batch_size) * (kwargs["v_stddev"] if rotate else 0) + kwargs["v_mean"]
        r_rotation = torch.zeros_like(h_rotation)

        return self.forward_with_rotation(data, h_rotation, v_rotation, r_rotation, **kwargs)


    @torch.no_grad()
    def forward_with_rotation(self, data, h_rotation, v_rotation, r_rotation, **kwargs):

        if self.mode == "fix_body":
            data, R_raster = self._forward_fix_body(data, h_rotation, v_rotation, r_rotation, **kwargs)
        elif self.mode in "fix_camera":
            data, R_raster = self._forward_fix_camera(data, h_rotation, v_rotation, r_rotation, **kwargs)
        else:
            return NotImplementedError

        data = self._forward_rasterize(data, R_raster, **kwargs)

        return data


    @torch.no_grad()
    def _forward_fix_body(self, data, h_rotation, v_rotation, r_rotation, **kwargs):

        # start = time.time()

        batch_size = data["scales"].shape[0]

        # print("preprocessor handling {} cases".format(batch_size))

        root_rotation = data["full_pose"][:, 0]

        euler = torch.zeros([batch_size, 3], device=self.device)
        euler[:, 1] = -h_rotation
        euler[:, 0] = math.pi - v_rotation
        euler[:, 2] = -r_rotation
        R = euler_angles_to_matrix(euler, convention="XYZ")
        R = root_rotation @ R
        R_raster = torch.inverse(R)

        # camera matrices
        body_rotation = F.pad(R, (0, 1, 0, 1), mode="constant", value=0.)
        body_rotation[:, -1, -1] = 1.
        world2cam_matrices = torch.bmm(torch.bmm(data["R"], data["T"]), body_rotation)
        cam2world_matrices = torch.inverse(world2cam_matrices.float())
        data["cam2world_matrices"] = cam2world_matrices

        return data, R_raster


    @torch.no_grad()
    def _forward_fix_camera(self, data, h_rotation, v_rotation, r_rotation, **kwargs):

        batch_size = data["scales"].shape[0]

        # print("preprocessor handling {} cases".format(batch_size))

        euler = torch.zeros([batch_size, 3], device=self.device)
        R_raster = euler_angles_to_matrix(euler, convention="XYZ")

        euler = torch.zeros([batch_size, 3], device=self.device)
        euler[:, 1] = h_rotation
        euler[:, 0] = v_rotation
        euler[:, 2] = r_rotation
        R = torch.eye(4, device=self.device, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1, 1)
        R[:, :3, :3] = euler_angles_to_matrix(euler, convention="XYZ")

        # fk
        tpose_vertices = data["tpose_vertices_shaped"]
        fk_matrices = data["fk_matrices"]
        lbs_weights = data["lbs_weights"]

        body_rotation = torch.inverse(R)
        fk_matrices = torch.einsum("bjk,bikl->bijl", body_rotation, fk_matrices)
        vertice_fk_matrices = torch.einsum("bji,bikl->bjkl", lbs_weights, fk_matrices)
        tpose_vertices_homo = F.pad(tpose_vertices, (0, 1), mode="constant", value=1.)
        vertices = torch.einsum("bijk,bik->bij", vertice_fk_matrices, tpose_vertices_homo)[..., :3]

        data["fk_matrices"] = fk_matrices
        data["vertices"] = vertices

        skeletons_xyz_homo = F.pad(data["skeletons_xyz"], (0, 1), mode="constant", value=1.)
        data["skeletons_xyz"] = torch.einsum("bjk,bik->bij", body_rotation, skeletons_xyz_homo)[..., :3]

        return data, R_raster


    @torch.no_grad()
    def _forward_rasterize(self, data, R_raster, **kwargs):

        batch_size = data["scales"].shape[0]

        faces = self.smpl_faces.unsqueeze(0).repeat(batch_size, 1, 1)
        meshes = Meshes(verts=data["vertices"], faces=faces).to(self.device)

        fov_raster = math.pi * 1 / 180
        focal_raster = 1. / math.tan(fov_raster / 2)
        T_raster = data["T"][:, :3, -1].clone()
        T_raster[:, -1] = focal_raster / data["scales"] * 0.5

        cameras = PerspectiveCameras(focal_length=-focal_raster, R=R_raster, T=T_raster, in_ndc=True,
                                     device=self.device)
        pix_to_face, zbuf, bary_coords, dists = self.rasterizer(meshes, cameras=cameras)

        # rasterize semantics

        pix_to_face = pix_to_face.reshape(batch_size, self.height, self.width)
        bg_mask = (pix_to_face < 0)
        pix_to_face = pix_to_face % len(self.smpl_faces)
        pix_to_face_verts = self.smpl_faces[pix_to_face]  # B, H, W, 3
        bary_coords = bary_coords[:, :, :, 0, :]  # B, H, W, 3
        pix_to_face_vert = torch.argmax(bary_coords, dim=-1, keepdim=True)
        pix_to_vert = torch.gather(pix_to_face_verts, dim=-1, index=pix_to_face_vert)
        pix_to_vert = pix_to_vert.reshape(batch_size, self.height, self.width)
        pix_to_vert[bg_mask] = -1

        rasterized_semantics = data["tpose_vertices"][0][pix_to_vert]
        rasterized_semantics[bg_mask.unsqueeze(-1).expand_as(rasterized_semantics)] = 0
        data["rasterized_semantics"] = rasterized_semantics.permute(0, 3, 1, 2)

        # rasterize segments
        rasterized_body_seg = self.smpl_faces_to_labels[pix_to_face] + 2
        rasterized_body_seg[bg_mask] = 1

        data["rasterized_segments"] = rasterized_body_seg

        return data


@torch.no_grad()
def get_preprocessor(dataloader, meta):

    preprocessor = SHHQPreprocessor(**meta)

    smpl = SMPL("./datasets/SMPL_NEUTRAL.pkl")
    smpl_faces = torch.from_numpy(smpl.faces.astype(np.int64))

    densepose_data_path = "./datasets/densepose_data.json"
    densepose_data = json.load(open(densepose_data_path))

    smpl_faces_to_labels = list(range(len(smpl_faces)))
    smpl_faces_to_labels = torch.tensor(densepose_data["smpl_faces_to_densepose_faces"], dtype=torch.long)[smpl_faces_to_labels]
    smpl_faces_to_labels = torch.tensor(densepose_data["densepose_faces_to_labels"], dtype=torch.long)[smpl_faces_to_labels]

    preprocessor.init_smpl(smpl_faces, smpl_faces_to_labels)

    return preprocessor
