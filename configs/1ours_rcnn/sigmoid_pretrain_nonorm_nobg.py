_base_ = 'base_rcnn_pretrain.py'

model = dict(
    roi_head=dict(
        type='IclipRoIHeadSigmoid',
        bbox_head=dict(
            type='IclipShared2FCBBoxHeadSigmoid',
            loss_cls=dict(
                _delete_=True, type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 / 100.
            )
        )
    )
)
