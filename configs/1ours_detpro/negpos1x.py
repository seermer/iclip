_base_ = 'base_rcnn.py'

model = dict(
    train_cfg=dict(
        rcnn=dict(
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=1.,
                num=512,
                pos_fraction=0.5,
                type='RandomSampler')
        )
    )
)
