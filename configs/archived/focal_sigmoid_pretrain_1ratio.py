_base_ = 'pretrain_1ratio.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                _delete_=True, type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
            )
        )
    )
)
