_base_ = 'base_rcnn_pretrain.py'

model = dict(
    roi_head=dict(
            bbox_head=dict(
                loss_cls=dict(
                    _delete_=True, type='FocalLoss', use_sigmoid=False, gamma=2.0, alpha=0.25, loss_weight=1.0)
            )
    )
)
