_base_ = 'sigmoid_pretrain_1ratio.py'

model = dict(
    roi_head=dict(
        type='IclipRoIHeadSigmoid',
        bbox_head=dict(
            type='IclipShared2FCBBoxHeadSigmoid',
        )
    )
)
