# Copyright (c) OpenMMLab. All rights reserved.
from pathlib import Path
from typing import List

import mmengine
from mmengine.dataset import BaseDataset
from mmengine.fileio import get_file_backend

from mmdet.registry import DATASETS


@DATASETS.register_module()
class IclipDataset(BaseDataset):
    def __init__(self,
                 *args,
                 backend_args=None,
                 **kwargs) -> None:

        super().__init__(*args, **kwargs)

    def load_data_list(self) -> List[dict]:
        """Load data list."""
        img_prefix = self.data_prefix['img']
        annotations = mmengine.load(self.ann_file)
        file_backend = get_file_backend(img_prefix)

        data_list = []
        for ann in annotations:
            if not (Path(self.data_root) / ann['image']).exists():
                continue

            data_info = {
                'img_id': Path(ann['image']).stem.split('_')[-1],
                'img_path': file_backend.join_path(img_prefix, ann['image']),
                'gt_caption': ann['caption'],
            }

            data_list.append(data_info)
        print(len(data_list))

        return data_list

