_base_ = 'faster-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'IclipDataset'
data_root = '/media/Auriga/fangyic/yfcc15m/'

img_scale = (128, 128)  # width, height

train_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(2, 11)),
    dict(type='RandomChoiceResize',
         scales=[(128, 128)],
         keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

custom_hooks = [dict(type='CheckInvalidLossHook', interval=1)]

train_dataset = dict(
    # use MultiImageMixDataset wrapper to support mosaic and mixup
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotation.json',
        data_prefix=dict(img='./'),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadExtractClipText',
                 text_encoder_model='RN50',
                 save_folder=data_root + 'capfeat/', init_clip=False, ann_file=data_root + 'annotation.json')
        ],
        filter_cfg=dict(filter_empty_gt=False),
        backend_args=None),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=256,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

model = dict(
    roi_head=dict(
        type='IclipRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='IclipShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1024,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))), )

max_iters = 102622  # 102622 is 1 epoch with batchsize 18*8  each iter == clip 43 epochs  18*8*102622 = 15M
# 102622 is 1/3 epoch with bs      6*8   each iter == clip 135 iter    6*8*102622  = 5M

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iters,
        by_epoch=False,
        milestones=[max_iters // 5 * 4],
        gamma=0.1)]

train_cfg = dict(
    _delete_=True,
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