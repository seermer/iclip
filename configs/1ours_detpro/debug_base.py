_base_ = 'faster-rcnn_r50_fpn_1x_coco.py'

dataset_type = 'IclipDataset'
data_root = '/media/Auriga/fangyic/yfcc15m/'

img_scale = (1024, 1024)  # width, height

train_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(2, 10), mode='rescalecentercrop'),
    dict(type='RandomChoiceResize',
         scales=[(608, 608), (640, 640), (672, 672), (704, 704),
                 (736, 763), (768, 768), (800, 1333), (832, 832),
                 (864, 864), (896, 896), (928, 928), (960, 960),
                 (992, 992), (1024, 1024)],
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
                 text_encoder_model='ViT-B/32',
                 save_folder=data_root + 'capfeat/', init_clip=False, ann_file=data_root + 'annotation.json')
        ],
        filter_cfg=dict(filter_empty_gt=False),
        backend_args=None),
    pipeline=train_pipeline)

train_dataloader = dict(
    batch_size=18,
    num_workers=5,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=train_dataset)

model = dict(
    # data_preprocessor=dict(
    #     type='DetDataPreprocessor',
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     bgr_to_rgb=True,
    #     pad_size_divisor=32),
    # backbone=dict(
    #     type='ResNet',
    #     depth=50,
    #     num_stages=4,
    #     out_indices=(0, 1, 2, 3),
    #     frozen_stages=-1,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     norm_eval=True,
    #     style='caffe',
    #     init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256, 512, 1024, 2048],
    #     out_channels=256,
    #     norm_cfg=dict(type='SyncBN', requires_grad=True),
    #     num_outs=5),
    roi_head=dict(
        type='IclipRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='IclipShared4Conv1FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=512,
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
        milestones=[max_iters // 12 * 8, max_iters // 12 * 11],
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

# train_cfg = dict(max_epochs=1, type='EpochBasedTrainLoop', val_interval=10000000000000)
val_cfg = None
val_dataloader = None
val_evaluator = None
test_cfg = None
test_dataloader = None
test_evaluator = None
