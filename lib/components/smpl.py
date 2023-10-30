import torch
import torch.nn.functional as F

from smplx.body_models import SMPL as _SMPL
from smplx.utils import Tensor, SMPLOutput
from smplx.lbs import blend_shapes, vertices2joints, batch_rodrigues, batch_rigid_transform
from typing import Tuple, List, Optional, Dict, Union
from pytorch3d.ops import knn_points, knn_gather


def lbs(
    betas: Tensor,
    pose: Tensor,
    v_template: Tensor,
    shapedirs: Tensor,
    posedirs: Tensor,
    J_regressor: Tensor,
    parents: Tensor,
    lbs_weights: Tensor,
    pose2rot: bool = True,
):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: torch.dtype, optional

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = torch.matmul(
            pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = torch.matmul(pose_feature.view(batch_size, -1),
                                    posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents, dtype=dtype)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)) \
        .view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1],
                               dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return A, v_shaped, verts, J, J_transformed


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(
            self,
            betas: Optional[Tensor] = None,
            body_pose: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts=True,
            return_full_pose: bool = False,
            pose2rot: bool = True,
            **kwargs
    ):
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        # If no shape and pose parameters are passed along, then use the
        # ones from the module
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas

        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl

        full_pose = torch.cat([global_orient, body_pose], dim=1)

        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])

        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)

        rigid_transforms, vertices_shaped, vertices, joints_shaped, joints = lbs(
            betas, full_pose, self.v_template,
            self.shapedirs, self.posedirs,
            self.J_regressor, self.parents,
            self.lbs_weights, pose2rot=pose2rot)

        joints = self.vertex_joint_selector(vertices, joints)
        # Map the joints to the current dataset
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)

        if apply_trans:
            joints += transl.unsqueeze(dim=1)
            vertices += transl.unsqueeze(dim=1)

        output = {
            "fk_matrices": rigid_transforms,
            "tpose_vertices": vertices_shaped if return_verts else None,
            "vertices": vertices if return_verts else None,
            "global_orient": global_orient,
            "body_pose": body_pose,
            "joints_shaped": joints_shaped,
            "joints": joints,
            "betas": betas,
            "full_pose": full_pose if return_full_pose else None,
            "lbs_weights": self.lbs_weights
        }

        return output


def get_geo_features(points, skeletons, vertices, tpose_vertices, fk_matrices, lbs_weights, legacy_mode=False):

    batch_size, n_points, _ = points.shape
    n_vertices = vertices.shape[1]

    joint_dists = torch.cdist(points, skeletons) / 2.4

    ik_matrices = torch.inverse(fk_matrices.float())
    vertex_ik_matrices = torch.einsum("bij,bjkl->bikl", lbs_weights, ik_matrices)

    nearest_dists, nearest_indices, _ = knn_points(points.float(), vertices.float())

    point_ik_matrices = knn_gather(vertex_ik_matrices.view(batch_size, n_vertices, 16), nearest_indices)
    point_ik_matrices = point_ik_matrices.mean(dim=2)
    point_ik_matrices = point_ik_matrices.reshape(batch_size, n_points, 4, 4)

    points_homo = F.pad(points, [0, 1], "constant", 1.)
    cano_points = torch.einsum("bijk,bik->bij", point_ik_matrices, points_homo)
    cano_points = cano_points[:, :, :3]
    cano_points[..., 0] = cano_points[..., 0] / 2.
    cano_points[..., 1] = (cano_points[..., 1] + 0.2) / 2.
    cano_points[..., 2] = cano_points[..., 2] / 1.3

    cano_vertices = knn_gather(tpose_vertices, nearest_indices[:, :, :1]).squeeze(2)
    cano_vertices[:, :, 1] = cano_vertices[:, :, 1]
    cano_vertices[:, :, 2] = cano_vertices[:, :, 2] / 0.2

    nearest_dists = torch.sqrt(nearest_dists[:, :, :1]) / 1.3

    if legacy_mode:
        geo_features = torch.cat([joint_dists, cano_points, cano_vertices, nearest_dists], dim=-1)
    else:
        geo_features = torch.cat([cano_points, joint_dists, cano_vertices, nearest_dists], dim=-1)

    # print()
    # for i in range(31):
    #     print("geo_features dim {}: min={:.3f}, mean={:.3f}, max={:.3f}".format(
    #         i, geo_features[..., i].min(), geo_features[..., i].mean(), geo_features[..., i].max()))

    return geo_features

