_base_ = 'base_rcnn_pretrain.py'

model = dict(
    roi_head=dict(
        type='IclipRoIHeadSigmoid',
        bbox_head=dict(
            type='IclipShared2FCBBoxHeadSigmoid',
            loss_cls=dict(
                _delete_=True, type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=2.0
            )
        )
    )
)

max_iters = 102622

param_scheduler = [
    dict(
        _delete=True, type='LinearLR', start_factor=0.00025, by_epoch=False, begin=0, end=2000),
    dict(
        _delete=True,
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[max_iters // 5 * 4],
        gamma=0.1)
]
