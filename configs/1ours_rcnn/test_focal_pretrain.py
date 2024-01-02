_base_ = 'focal_pretrain.py'

dataset_type = 'IclipDataset'
data_root = '/media/Auriga/fangyic/yfcc15m/'

img_scale = (1024, 1024)  # width, height

model = dict(
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5, class_agnostic=True),
            max_per_img=100))
)

val_pipeline = [
    dict(type='Collage', img_scale=img_scale, grid_range=(2, 9), mode='rescalecentercrop'),
    dict(type='RandomChoiceResize',
         scales=[img_scale],
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
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=val_dataset)

val_cfg = dict(_delete_=True, type='ValLoop')
test_cfg = dict(_delete_=True, type='TestLoop')
val_dataloader = val_dataloader
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='IclipMetric',
    ann_file=None,
    metric='bbox',
    format_only=False,
    backend_args=None)
test_evaluator = val_evaluator
