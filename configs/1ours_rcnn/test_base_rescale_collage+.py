_base_ = 'faster-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'IclipDataset'
data_root = '/media/Auriga/fangyic/yfcc15m/'

img_scale = (2048, 2048)  # width, height

train_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(5, 18), mode='rescalecentercrop'),
    dict(type='RandomChoiceResize',
         scales=[(1763, 1763), (1833, 1833),
                 (1896, 1896), (1928, 1928), (1960, 1960),
                 (1992, 1992), (2048, 2048)],
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
    batch_size=6,
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
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),
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

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=5000,
        max_keep_ckpts=3))

val_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(5, 18), mode='rescalecentercrop'),
    dict(type='RandomChoiceResize',
         scales=[(2048, 2048)],
         keep_ratio=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]
val_dataset = dict(
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
    pipeline=val_pipeline)
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

val_cfg = dict(type='ValLoop')
test_cfg = dict(
    rcnn=dict(
        max_per_img=100,
        nms=dict(iou_threshold=0.5, type='nms'),
        score_thr=0.05),
    rpn=dict(
        max_per_img=1000,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.7, type='nms'),
        nms_pre=1000))
val_dataloader = val_dataloader
test_dataloader = val_dataloader

val_evaluator = dict(
    type='IclipMetric',
    ann_file=None,
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator
