_base_ = 'focal_pretrain.py'

model = dict(
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=0.5,
                num=512,
                pos_fraction=0.75,
                type='RandomSampler'))))
