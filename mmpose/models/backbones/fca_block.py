import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmpose.registry import MODELS

from mmpose.models.necks.FCANet import FcaBasicBlock, FcaBottleneck  # 你已有的 block 实现
from torchvision.models.resnet import conv1x1


@MODELS.register_module()
class FcaResNetBackbone(BaseModule):
    """FcaNet Backbone for OpenMMLab (resnet-style).

    Args:
        depth (int): Network depth. Options: 34, 50, 101, 152.
        in_channels (int): Input image channels. Default: 3
        out_indices (tuple[int]): Indices of output feature maps (e.g. (0, 1, 2, 3)).
        frozen_stages (int): Number of frozen stages. Default: -1 (no freeze)
    """

    arch_settings = {
        34: (FcaBasicBlock, [3, 4, 6, 3]),
        50: (FcaBottleneck, [3, 4, 6, 3]),
        101: (FcaBottleneck, [3, 4, 23, 3]),
        152: (FcaBottleneck, [3, 8, 36, 3])
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 norm_eval=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        block, layers = self.arch_settings[depth]
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        for i, layer in enumerate([self.layer1, self.layer2, self.layer3, self.layer4]):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        # freeze layers
        if self.frozen_stages >= 0:
            self.conv1.eval()
            self.bn1.eval()
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
