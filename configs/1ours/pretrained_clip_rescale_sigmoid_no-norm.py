_base_ = 'pretrained_clip_rescale.py'


model = dict(bbox_head=dict(type='IclipDeformableDETRHead3'))

