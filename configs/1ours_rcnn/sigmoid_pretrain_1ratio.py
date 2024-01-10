_base_ = 'pretrain_1ratio.py'

grid = list(range(2, 10))
weight = sum(x ** 2 for x in grid) / len(grid)

model = dict(
    roi_head=dict(
        bbox_head=dict(
            loss_cls=dict(
                _delete_=True, type='CrossEntropyLoss', use_sigmoid=True, loss_weight=weight
            )
        )
    )
)
