_base_ = 'base_rcnn_pretrain.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            type='IclipShared2FCBBoxHeadSigmoid',
            loss_cls=dict(
                _delete_=True, type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,
                loss_weight=2.0
            )
        )
    )
)
