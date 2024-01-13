# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch import nn

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import DetDataSample, SampleList
from mmdet.structures.bbox import bbox2roi
from mmdet.utils import ConfigType, InstanceList
from ..task_modules.samplers import SamplingResult
from ..utils import empty_instances, unpack_gt_instances
from mmdet.models.roi_heads.iclip_roi_head import IclipRoIHead
from mmdet.utils.logger import print_log


@MODELS.register_module()
class IclipRoIHead2(IclipRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self, ensemble=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bg_embedding = nn.Linear(1, 512)
        nn.init.xavier_uniform_(self.bg_embedding.weight)
        nn.init.constant_(self.bg_embedding.bias, 0)

        self.projection = nn.Linear(1024, 512)
        self.temperature = 0.01
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

    def _bbox_forward(self, x: Tuple[Tensor], rois: Tensor, batch_data_samples) -> dict:
        """Box head forward function used in both training and testing.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rois (Tensor): RoIs with the shape (n, 5) where the first
                column indicates batch id of each RoI.

        Returns:
             dict[str, Tensor]: Usually returns a dictionary with keys:

                - `cls_score` (Tensor): Classification scores.
                - `bbox_pred` (Tensor): Box energies / deltas.
                - `bbox_feats` (Tensor): Extract bbox RoI features.
        """
        caption_feat = []
        idx_wrapper = 0
        for data_sample in batch_data_samples:
            if min(data_sample.gt_instances['labels']) < idx_wrapper:
                data_sample.gt_instances['labels'] += idx_wrapper  # align the pseudo label with caption idx
            idx_wrapper += len(data_sample.gt_instances['capfeats'])
            caption_feat.append(data_sample.gt_instances['capfeats'])
        caption_feat_all_GPU, gt_per_img = self.gather_all_capfeat(caption_feat)
        self.bbox_head.num_classes = len(caption_feat_all_GPU)

        input_one = x[0].new_ones(1)
        bg_class_embedding = self.bg_embedding(input_one).reshape(1, 512)
        bg_class_embedding = torch.nn.functional.normalize(bg_class_embedding, p=2, dim=1)
        caption_feat_all_GPU = torch.cat([caption_feat_all_GPU, bg_class_embedding], dim=0)

        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        bbox_pred = self.bbox_head(bbox_feats)
        region_embeddings_image = self.projection(self.bbox_head.forward_embedding(bbox_feats))
        region_embeddings_image = torch.nn.functional.normalize(region_embeddings_image, p=2, dim=1)
        cls_score = region_embeddings_image @ caption_feat_all_GPU.T
        cls_score /= self.temperature
        print_log(f'DEBUG CLS_SCORE {bbox_pred.shape, region_embeddings_image.shape, caption_feat_all_GPU.shape, cls_score.shape}')

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats)
        return bbox_results
