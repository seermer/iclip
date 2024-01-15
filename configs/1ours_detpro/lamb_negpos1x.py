_base_ = 'negpos1x.py'

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(_delete_=True, type='Lamb', lr=0.02, betas=(0.9, 0.999), weight_decay=0.0001))
