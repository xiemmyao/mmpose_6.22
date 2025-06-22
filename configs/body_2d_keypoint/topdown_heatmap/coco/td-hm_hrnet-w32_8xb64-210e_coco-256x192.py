custom_imports = dict(
    imports=['mmpose.models.backbones.hrnet'],
    allow_failed_imports=False)
_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=5e-4,
))
# optim_wrapper = dict(
#     optimizer=dict(
#         type='Adam',
#         lr=5e-4,  # 学习率，根据你的实际需求调整
#         weight_decay=0.0001  # 权重衰减率
#     ),
#     paramwise_cfg=dict(
#         custom_keys={
#             'backbone.layer0': dict(lr_mult=0, decay_mult=0),
#             'backbone.layer1': dict(lr_mult=0, decay_mult=0),
#         }
#     )
# )

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True)
]


# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# codec settings
codec = dict(
    type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# model settings
model = dict(
    type='TopdownPoseEstimator',
    # 数据预处理
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,#第一阶段有一个模块
                num_branches=1,#该阶段有一个分支。
                block='BOTTLENECK',#使用 Bottleneck 结构作为块类型，Bottleneck 是一种常用于 ResNet 等网络的结构，用于提高网络的表达能力
                num_blocks=(4, ),#每个分支包含4个基本块
                num_channels=(64, )),#该阶段的分支输出通道数为64
            stage2=dict(
                num_modules=1,
                num_branches=2,#该阶段的分支输出通道数为64
                block='BASIC',
                num_blocks=(4, 4),#每个分支包含4个基本块
                num_channels=(32, 64)),#第一个分支输出32个通道，第二个分支输出64个通道
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256))),
        init_cfg=None
        # dict(
        #     type='Pretrained',
        #     checkpoint='https://download.openmmlab.com/mmpose/'
        #     'pretrain_models/hrnet_w32-36af842e.pth'),

    ),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

# pipelines
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# data loaders
train_dataloader = dict(
    # batch_size=64,原来的
    batch_size=4,
    # num_workers=2,原来的
    num_workers=0,
    # persistent_workers=True,原来的
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline,
    ))
val_dataloader = dict(
    # batch_size=32,原来
    batch_size=2,
    # num_workers=2,原来的
    num_workers=0,
    # persistent_workers=True,原来的
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        # bbox_file='data/coco/person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator


# visualizer = dict(vis_backends=[
#     dict(type='LocalVisBackend'),
#     dict(type='TensorboardVisBackend'),
# ])
