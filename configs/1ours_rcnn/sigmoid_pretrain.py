_base_ = 'base_rcnn_pretrain.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                _delete_=True, type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0 / 100.
            )
        )
    )
)
