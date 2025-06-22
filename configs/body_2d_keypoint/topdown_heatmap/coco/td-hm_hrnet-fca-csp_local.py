custom_imports = dict(
    imports=['mmpose.models.backbones.hrnet'],  # 引入你自定义 hrnet.py
    allow_failed_imports=False)

_base_ = ['../../../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=1, val_interval=1)  # 只训练1个epoch

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
)

# learning policy
param_scheduler = [
    dict(type='LinearLR', begin=0, end=100, start_factor=0.001, by_epoch=False),
    dict(type='MultiStepLR', begin=0, end=1, milestones=[1], gamma=0.1, by_epoch=True)
]

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# heatmap 编码器设置
codec = dict(type='MSRAHeatmap', input_size=(192, 256), heatmap_size=(48, 64), sigma=2)

# 模型结构（简化版 HRNet）
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(num_modules=1, num_branches=1, block='BOTTLENECK', num_blocks=(1,), num_channels=(32,)),
            stage2=dict(num_modules=1, num_branches=2, block='BASIC', num_blocks=(1, 1), num_channels=(16, 32)),
            stage3=dict(num_modules=1, num_branches=3, block='BASIC', num_blocks=(1, 1, 1), num_channels=(16, 32, 64)),
            stage4=dict(num_modules=1, num_branches=4, block='BASIC', num_blocks=(1, 1, 1, 1), num_channels=(16, 32, 64, 128))
        ),
        init_cfg=None
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=16,  # 对应 stage4 第一个分支
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=False,
        flip_mode='heatmap',
        shift_heatmap=False)
)

# dataset 配置
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'

train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal', prob=0.5),
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

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_train2017.json',
        data_prefix=dict(img='train2017/'),
        pipeline=train_pipeline
    )
)
val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='annotations/person_keypoints_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=val_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='CocoMetric', ann_file=data_root + 'annotations/person_keypoints_val2017.json')
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=512)
