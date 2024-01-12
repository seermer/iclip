# training schedule for 1x
max_iters = 102622  # 102622 is 1 epoch with batchsize 18*8  each iter == clip 43 epochs  18*8*102622 = 15M
# 102622 is 1/3 epoch with bs      6*8   each iter == clip 135 iter    6*8*102622  = 5M

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[max_iters // 12 * 8, max_iters // 12 * 11],
        gamma=0.1)]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=10000000000000)

# train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=10000000000000)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
