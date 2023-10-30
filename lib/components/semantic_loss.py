import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.data.utils import print_stats


class SemanticLoss(nn.Module):

    def __init__(self, semantic_dim, **kwargs):
        super().__init__()

        self.semantic_dim = semantic_dim
        self.x = nn.Linear(1, 1)
        self.register_buffer("smpl_geodist_prior", torch.zeros([semantic_dim, semantic_dim], dtype=torch.float))

    @torch.no_grad()
    def init(self, smpl_geodist_prior):
        self.smpl_geodist_prior.copy_(smpl_geodist_prior)

    def forward(self, pred, gt, max_batch_size=8, **kwargs):
        """
        pred: batch_size, num_vertices, height, width
        gt: batch_size, height, width
        """

        B, _, H, W = pred.shape
        max_batch_size = min(max_batch_size, B)

        if gt.shape[1] != H or gt.shape[2] != W:
            with torch.no_grad():
                gt = gt.unsqueeze(1).float()
                gt = F.interpolate(gt, (H, W), mode="nearest")
                gt = gt.squeeze(1).long()

        loss = 0
        score = 0

        for start in range(0, B, max_batch_size):
            end = start + max_batch_size

            gt_batch = gt[start:end].view(max_batch_size, H * W)
            pred_batch = pred[start:end].view(max_batch_size, self.semantic_dim, H * W)
            gt_soft = self.smpl_geodist_prior[gt_batch].permute(0, 2, 1)
            nll = (-F.log_softmax(pred_batch, dim=1) * gt_soft).sum(dim=1)

            # if torch.any(gt_batch > 0):
            #     gt_one_hot = F.one_hot(gt_batch, num_classes=self.semantic_dim).permute(0, 2, 1)
            #     class_occurence = torch.sum(gt_one_hot, dim=(0, 2))
            #     class_occurence[0] = 0
            #     num_classes_occur = (class_occurence > 0).sum()
            #     coefficients = torch.reciprocal(class_occurence.float()) * torch.numel(gt_soft) / \
            #                    (num_classes_occur * gt_soft.shape[1])
            #     coefficients[torch.isinf(coefficients)] = 0
            #     weight_map = coefficients[gt_batch]
            #     print_stats(weight_map.unsqueeze(-1), "weight_map")
            #     nll = nll * weight_map

            if torch.any(gt_batch > 0):
                bg_mask = (gt_batch == 1)
                bg_occurence = torch.count_nonzero(bg_mask).float()
                weight_map = torch.zeros_like(nll)
                weight_map[bg_mask] = torch.reciprocal(bg_occurence) * torch.numel(gt_soft) / (self.semantic_dim ** 2)
                weight_map[~bg_mask] = torch.reciprocal((torch.numel(gt_batch) - bg_occurence)) * torch.numel(gt_soft) / self.semantic_dim
                weight_map[torch.isinf(weight_map)] = 0
                nll = nll * weight_map

            loss += nll.mean()

            with torch.no_grad():
                pred_labels = torch.argmax(pred_batch[:, 1:, :], dim=1) + 1
                scores = torch.gather(gt_soft, dim=1, index=pred_labels.unsqueeze(1))
                score += scores.mean()

        loss = loss / min(B // max_batch_size, 1)

        with torch.no_grad():
            score = score / min(B // max_batch_size, 1)
            pred = pred.view(B, self.semantic_dim, H * W)
            real_prob = (1 - torch.softmax(pred, dim=1)).mean()

        return loss, score, real_prob
