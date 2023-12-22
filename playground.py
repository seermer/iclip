import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union
import os, torch, clip
import cv2, json
import mmcv
import numpy
from tqdm import tqdm
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random
from pathlib import Path

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale


class Collage(BaseTransform):

    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 grid_range: Tuple[int, int] = (2, 11),
                 mode='resize') -> None:
        assert isinstance(img_scale, tuple)
        self.grid_range = grid_range
        log_img_scale(img_scale, skip_square=True, shape_order='wh')
        self.img_scale = img_scale
        self.mode = mode

    @cache_randomness
    def get_indexes(self, dataset: BaseDataset) -> int:
        self.n = 5
        indexes = [random.randint(0, len(dataset)) for _ in range(self.n ** 2 - 1)]
        return indexes

    def patch_proc(self, img, size):
        if self.mode == 'resize':
            return mmcv.imresize(img, size)
        elif self.mode == 'rescalecentercrop':
            img = mmcv.imrescale(img, (size[0], 1e6))
            img_height, img_width = img.shape[:2]

            crop_height = size[1]
            crop_width = size[0]
            y1 = max(0, int(round((img_height - crop_height) / 2.)))
            x1 = max(0, int(round((img_width - crop_width) / 2.)))
            y2 = min(img_height, y1 + crop_height) - 1
            x2 = min(img_width, x1 + crop_width) - 1
            bboxes = np.array([x1, y1, x2, y2])

            img = mmcv.imcrop(img, bboxes=bboxes)
            return img
        else:
            raise AttributeError

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        assert 'mix_results' in results

        result_patch = copy.deepcopy(results)
        others_patch = copy.deepcopy(results['mix_results'])
        sub_img_wh = self.img_scale[0] // self.n

        for i in range(self.n):
            if i == 0:  # first row
                img_row = self.patch_proc(result_patch['img'], (sub_img_wh, sub_img_wh))
                gt_bboxes = np.array([[sub_img_wh // 2, sub_img_wh // 2]], dtype=np.float32)
                for other_patch in others_patch[0:self.n - 1]:
                    img_o = self.patch_proc(other_patch['img'], (sub_img_wh, sub_img_wh))
                    img_row = np.concatenate((img_row, img_o), axis=1)
                    gt_bboxes = np.concatenate((gt_bboxes, gt_bboxes[-1].reshape((1, 2)) + [sub_img_wh, 0]), axis=0)
                img_col = img_row
                gt_bboxes = np.expand_dims(gt_bboxes, axis=0)
            else:
                img_row = None
                for other_patch in others_patch[i * self.n - 1: (i + 1) * self.n - 1]:
                    img_o = self.patch_proc(other_patch['img'], (sub_img_wh, sub_img_wh))
                    img_row = np.concatenate((img_row, img_o), axis=1) if img_row is not None else img_o
                img_col = np.concatenate((img_col, img_row), axis=0)
                gt_bboxes = np.concatenate((gt_bboxes, gt_bboxes[-1, :].reshape((1, -1, 2)) + [[0, sub_img_wh]]),
                                           axis=0)
        results['img'] = img_col
        results['img_shape'] = img_col.shape[:2]

        results['gt_bboxes'] = self.get_bboxes_target(gt_bboxes, sub_img_wh)
        print(results['gt_bboxes'], 1)
        results['gt_bboxes_labels'] = self.get_pseudo_label(gt_bboxes)

        # caption_feat = [results['capfeat']]
        # for i in results['mix_results']:
        #     caption_feat.append(i['capfeat'])
        # results['capfeat'] = torch.cat(caption_feat, dim=0)
        return results

    @autocast_box_type()
    def new_transform(self, results: dict) -> dict:
        sub_img_wh = self.img_scale[0] // self.n
        out_img_wh = sub_img_wh * self.n
        sub_img_shape = (sub_img_wh, sub_img_wh)
        collage_lst = [self.patch_proc(results['img'], sub_img_shape)]
        collage_lst.extend(self.patch_proc(img['img'], sub_img_shape)
                           for img in results['mix_results'][:self.n * self.n - 1])

        collage_arr = np.empty((out_img_wh, out_img_wh, collage_lst[0].shape[-1]), dtype=collage_lst[0].dtype)
        for i, img in enumerate(collage_lst):
            row, col = i // self.n * sub_img_wh, i % self.n * sub_img_wh
            collage_arr[row:row + sub_img_wh, col:col + sub_img_wh] = img

        results['img'] = collage_arr
        results['img_shape'] = collage_arr.shape[:2]

        bbox_tl = np.tile(np.arange(0, self.n * sub_img_wh, sub_img_wh), (self.n, 1))
        bbox_x1 = bbox_tl.flatten()
        bbox_y1 = bbox_tl.T.flatten()
        sub_img_wh = sub_img_wh // 2 * 2
        bbox_x2 = bbox_x1 + sub_img_wh
        bbox_y2 = bbox_y1 + sub_img_wh

        gt_bboxes = np.stack([bbox_x1, bbox_y1, bbox_x2, bbox_y2]).T
        print(gt_bboxes)

        results['gt_bboxes'] = gt_bboxes
        results['gt_bboxes_labels'] = np.arange(len(gt_bboxes))

        # caption_feat = [results['capfeat']]
        # for i in results['mix_results']:
        #     caption_feat.append(i['capfeat'])
        # results['capfeat'] = torch.cat(caption_feat, dim=0)
        return results

    def get_pseudo_label(self, bboxes):
        n = len(bboxes.reshape(-1, 2))
        return np.arange(n)

    def get_bboxes_target(self, bboxes, sub_img_wh):
        locat_info = bboxes.reshape(-1, 2)
        scale_info = np.array([[sub_img_wh, sub_img_wh]])
        n = locat_info.shape[0]
        expanded_array = np.tile(scale_info, (n, 1))
        res = np.concatenate((locat_info, expanded_array), axis=1).astype(np.float32)
        return self.cxcywh_to_xyxy(res)

    def cxcywh_to_xyxy(self, xcycwh_bboxes):
        xyxy_bboxes = xcycwh_bboxes.copy()
        xyxy_bboxes[:, 0] = xcycwh_bboxes[:, 0] - xcycwh_bboxes[:, 2] // 2  # Calculate X1
        xyxy_bboxes[:, 1] = xcycwh_bboxes[:, 1] - xcycwh_bboxes[:, 3] // 2  # Calculate Y1
        xyxy_bboxes[:, 2] = xcycwh_bboxes[:, 0] + xcycwh_bboxes[:, 2] // 2  # Calculate X2
        xyxy_bboxes[:, 3] = xcycwh_bboxes[:, 1] + xcycwh_bboxes[:, 3] // 2  # Calculate Y2
        return xyxy_bboxes


def get_cmap(n=256):
    # adapted from Pascal VOC segmentation coloring
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype=np.uint8)
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])
    return cmap


def main():
    n = 12
    from copy import deepcopy
    cmap = get_cmap()
    results = {
        'img': np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8),
        'mix_results': [{'img': np.ones((400, 400, 3), dtype=np.uint8) * cmap[i].reshape(1, 1, 3)}
                        for i in range(n * n)]
    }
    res1 = deepcopy(results)
    res2 = deepcopy(results)

    t = Collage((1024, 1024))
    t.get_indexes(list(range(1000)))
    t.transform(res1)
    t.new_transform(res2)

    for k, v in res1.items():
        if k == 'mix_results':
            continue
        print(k)
        if isinstance(v, np.ndarray):
            if isinstance(v == res2[k], bool):
                print(v == res2[k], v.shape, res2[k].shape)
            else:
                print((v == res2[k]).all())
        else:
            print(v == res2[k])
        print()


if __name__ == '__main__':
    main()
