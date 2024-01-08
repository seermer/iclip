_base_ = 'focal_pretrain.py'

model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=0.5,
                num=512,
                pos_fraction=0.75,
                type='RandomSampler'))))
