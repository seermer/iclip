_base_ = 'lamb_negpos1x.py'

train_dataloader = dict(
    batch_size=36
)

optim_wrapper = dict(
    optimizer=dict(lr=0.04)
)
