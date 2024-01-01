_base_ = 'base_rcnn_pretrain.py'

model = dict(
    roi_head=dict(
        type='IclipRoIHeadSigmoid',
        bbox_head=dict(
            type='IclipShared2FCBBoxHeadSigmoid',
            loss_cls=dict(
                _delete_=True, type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
        )
    )
)

max_iters = 102622

param_scheduler = [
    dict(type='LinearLR', start_factor=0.00025, by_epoch=False, begin=0, end=2000),
    dict(type='MultiStepLR',
         begin=0,
         end=max_iters,
         by_epoch=False,
         milestones=[max_iters // 5 * 4],
         gamma=0.1)
]
