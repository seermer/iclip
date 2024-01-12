# Copyright (c) OpenMMLab. All rights reserved.
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .multi_instance_bbox_head import MultiInstanceBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .iclip_bbox_head import IclipBBoxHead
from .iclip_bbox_head_sigmoid import IclipBBoxHeadSigmoid
from .iclip_convfc_bbox_head import IclipShared4Conv1FCBBoxHead, IclipShared2FCBBoxHead, IclipConvFCBBoxHead
from .iclip_convfc_bbox_head_sigmoid import (IclipShared4Conv1FCBBoxHeadSigmoid, IclipShared2FCBBoxHeadSigmoid,
                                             IclipConvFCBBoxHeadSigmoid)

from .iclip_convfc_bbox_head2 import IclipShared4Conv1FCBBoxHead2  # vild like

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'MultiInstanceBBoxHead',
    'IclipBBoxHead', 'IclipShared4Conv1FCBBoxHead', 'IclipShared2FCBBoxHead', 'IclipConvFCBBoxHead',
    'IclipShared4Conv1FCBBoxHeadSigmoid', 'IclipShared2FCBBoxHeadSigmoid', 'IclipConvFCBBoxHeadSigmoid',
    'IclipShared4Conv1FCBBoxHead2'
]
