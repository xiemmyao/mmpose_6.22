import torch
from mmengine.config import Config
import sys
import os

# 添加 mmpose 路径，使其可导入
sys.path.append(os.path.join(os.path.dirname(__file__), 'mmpose'))

# 从模型定义路径导入 HRNet
from mmpose.models.backbones.hrnet import HRNet

if __name__ == '__main__':
    # 构建简化配置
    cfg = Config(dict(
        model=dict(
            type='HRNet',
            extra=dict(
                stage1=dict(
                    num_modules=1,
                    num_branches=1,
                    block='BOTTLENECK',
                    num_blocks=[4],
                    num_channels=[64]
                ),
                stage2=dict(
                    num_modules=1,
                    num_branches=2,
                    block='BASIC',
                    num_blocks=[4, 4],
                    num_channels=[32, 64]
                ),
                stage3=dict(
                    num_modules=1,
                    num_branches=3,
                    block='BASIC',
                    num_blocks=[4, 4, 4],
                    num_channels=[32, 64, 128]
                ),
                stage4=dict(
                    num_modules=1,
                    num_branches=4,
                    block='BASIC',
                    num_blocks=[4, 4, 4, 4],
                    num_channels=[32, 64, 128, 256],
                    multiscale_output=True
                )
            )
        )
    ))

    # 构建模型
    model = HRNet(extra=cfg.model['extra'])

    # 构造测试输入
    input_tensor = torch.randn(1, 3, 256, 192)  # Batch size of 1, 3 channels, 256x192 resolution

    # 前向传播测试
    with torch.no_grad():
        output = model(input_tensor)

    print("\n✅ Forward pass successful!")
    # for i, out in enumerate(output):
    #     print(f"Output[{i}] shape: {out.shape}")
