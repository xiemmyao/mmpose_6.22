# Copyright (c) OpenMMLab. All rights reserved.
import copy
import torch.nn as nn
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmengine.model import BaseModule, constant_init
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.registry import MODELS
from .base_backbone import BaseBackbone
from .resnet import BasicBlock, Bottleneck, get_expansion
from mmengine.model import BaseModule

from mmpose.models.necks import CSPNeXtPAFPN
from mmpose.models.necks.FCANet import MultiSpectralAttentionLayer




class HRModule(BaseModule):

    def __init__(self,
                 num_branches,#分支的数量。HRNet 通常使用多个分支，每个分支处理不同分辨率的特征，允许模型在多个尺度上进行信息融合。
                 blocks,#每个分支中使用的基本块（block）类型，可能是 BasicBlock 或 Bottleneck，用于构建每个分支
                 num_blocks,#每个分支中包含的块数
                 in_channels,
                 num_channels,#每个分支的输出通道数，是一个列表，表示每个分支的通道数
                 multiscale_output=False,#是否启用多尺度输出
                 with_cp=False,#是否启用计算图重用（checkpointing），这可以在训练时节省内存，但可能会稍微增加计算开销
                 conv_cfg=None,#卷积配置，通常包含卷积的具体参数
                 norm_cfg=dict(type='BN'),#归一化层的配置，默认为 BatchNorm（BN）
                 upsample_cfg=dict(mode='nearest', align_corners=None),#上采样配置，默认使用 nearest 模式进行上采样
                 init_cfg=None):#模型初始化配置，可能包括预训练模型的加载等


        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)#调用父类 BaseModule 的初始化方法，初始化模块时传入预定义的 init_cfg 配置。BaseModule 是该模块的父类，通常包含一些通用的方法和属性，比如网络参数的初始化、模型保存和加载等
        self._check_branches(num_branches, num_blocks, in_channels,
                             num_channels)#方法用来检查分支配置是否有效

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.multiscale_output = multiscale_output
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.upsample_cfg = upsample_cfg
        self.with_cp = with_cp
        self.branches = self._make_branches(num_branches, blocks, num_blocks,
                                            num_channels)
        self.fuse_layers = self._make_fuse_layers()#这个方法负责构建各个分支之间的融合层。在 HRNet 中，不同分支提取的特征需要通过融合层进行合并，以形成最终的高分辨率特征图。融合的方式可以是加权求和、拼接等
        self.relu = nn.ReLU(inplace=True)

        # 添加两个模块用于融合后输出
        self.csp_pafpn = CSPNeXtPAFPN(
            in_channels=self.in_channels,
            out_channels=min(self.in_channels),
            out_indices=tuple(range(len(self.in_channels)))
        )
        self.fca_modules = nn.ModuleList([
            MultiSpectralAttentionLayer(
                channel=c,
                dct_h=h,
                dct_w=w,
                reduction=16,
                freq_sel_method='top16'
            ) for c, h, w in zip(self.in_channels,
                                 [64, 32, 16, 8][:num_branches],
                                 [48, 24, 12, 6][:num_branches])
        ])

    @staticmethod
    def _check_branches(num_branches, num_blocks, in_channels, num_channels):#这个方法是一个静态方法，它的主要作用是验证给定的分支配置是否有效，确保分支的数量与其它相关配置项一致
        """Check input to avoid ValueError."""
        if num_branches != len(num_blocks):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_BLOCKS({len(num_blocks)})'
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_CHANNELS({len(num_channels)})'
            raise ValueError(error_msg)

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_one_branch(self,#构建单个分支
                         branch_index,#当前分支的索引，指示要为哪个分支创建计算图
                         block,#当前分支使用的基本块类型（例如，BasicBlock 或 Bottleneck）
                         num_blocks,
                         num_channels,
                         stride=1):
        """Make one branch."""
        downsample = None
#下采样（如果需要）：如果当前的输入通道数与目标输出通道数不匹配，或者需要下采样（步长 stride 不为 1），则先使用一个 1x1 卷积进行下采样。
        if stride != 1 or \
                self.in_channels[branch_index] != \
                num_channels[branch_index] * get_expansion(block):
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(
                    self.norm_cfg,
                    num_channels[branch_index] * get_expansion(block))[1])
        # 添加基本块：为当前分支添加若干个基本块（BasicBlock 或 Bottleneck），每个块处理特征图并更新通道数。
        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index] * get_expansion(block),
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        self.in_channels[branch_index] = \
            num_channels[branch_index] * get_expansion(block)
        for _ in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index] * get_expansion(block),
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))
        # 返回分支：将所有的块按顺序组成一个 nn.Sequential 容器，表示当前分支的计算过程
        return nn.Sequential(*layers)
#构建多个分支
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)
#负责构建分支之间的融合层
    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None#分支数为 1 时：如果只有一个分支（即 self.num_branches == 1），则不需要任何融合层，直接返回 None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
#对于每一对分支（i 和 j），如果 j > i，表示分支 j 的特征图尺寸大于分支 i，需要通过上采样（Upsample）来调整 j 的特征图大小，以便与分支 i 的特征图进行融合。
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners'])))
                elif j == i:#如果 j == i，表示两个分支的特征图尺寸相同，不需要进行任何操作，因此返回 None。
                    fuse_layer.append(None)
                else:#如果 j < i，则需要进行下采样，通过卷积层来调整分支 j 的特征图大小，使其与分支 i 对齐。
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)
    # x 是一个包含多个张量的列表，每个张量代表一个不同分支的输入特征图。每个分支都负责处理不同分辨率的特征，因此 x 中每个元素的大小可能不同
    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
#网络的结构是单分支的，没有需要进行融合的操作，因此可以直接通过 self.branches[0](x[0]) 进行计算
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
#如果有多个分支（self.num_branches > 1），这段代码将每个分支的输入 x[i] 传入对应的分支 self.branches[i] 中进行处理
        x_fuse = []#这个列表用来存储每个融合后的特征图
        for i in range(len(self.fuse_layers)):#是一个包含所有分支之间融合操作的列表，长度等于分支的数量（如果是多尺度输出，则会有多个融合层）
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
         # 插入 CSPNeXtPAFPN（融合之后）
        x_fuse = list(self.csp_pafpn(tuple(x_fuse)))
        return x_fuse


@MODELS.register_module()#这是一个装饰器，用于将 HRNet 类注册到模型库中，使其可以通过框架的接口进行调用
class HRNet(BaseBackbone):#HRNet 类继承自 BaseBackbone，这意味着它是一个骨干网络（backbone），负责从原始图像中提取特征


    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck}

    def __init__(
        self,
        extra,#extra 是一个包含网络每个阶段配置的字典
        in_channels=3,
        conv_cfg=None,#用于配置卷积层的字典。如果为 None，则使用默认的卷积配置
        norm_cfg=dict(type='BN'),
        norm_eval=False,#是否在训练过程中将归一化层（如 BatchNorm）的 trainable 模式设置为 False
        with_cp=False,#是否使用检查点
        zero_init_residual=False,#是否使用零初始化（zero initialization）来初始化残差模块中的归一化层
        frozen_stages=-1,#指定冻结哪些阶段（stage）的参数。冻结意味着在训练过程中不更新这些阶段的权重。-1 表示不冻结任何参数。其他值表示冻结到指定的阶段索引。例如，如果设置为 2，则表示冻结第一个到第二个阶段的所有层（不更新权重）。
        init_cfg=[
            dict(type='Normal', std=0.001, layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],#这是一个初始化配置的列表或字典，用于定义如何初始化网络的各层


    ):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.zero_init_residual = zero_init_residual
        self.frozen_stages = frozen_stages


# #将传入的各种配置参数保存到实例变量中，以便后续使用
        # stem net
        self.norm1_name, norm1 = build_norm_layer(self.norm_cfg, 64, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(self.norm_cfg, 64, postfix=2)
#这部分代码定义了网络的 stem 部分（即网络的初始部分）
        self.conv1 = build_conv_layer(
            self.conv_cfg,
            in_channels,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)
#构建卷积层的辅助函数
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(#定义并构建第二个卷积层 conv2
            self.conv_cfg,
            64,
            64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

        self.add_module(self.norm2_name, norm2)
        self.relu = nn.ReLU(inplace=True)

        self.upsample_cfg = self.extra.get('upsample', {
            'mode': 'nearest',
            'align_corners': None
        })
#获取上采样的配置
        # stage 1
        #读取 extra 配置字典中的 stage1 配置，获取该阶段的相关参数（如输出通道数、使用的块类型、块数等）
        self.stage1_cfg = self.extra['stage1']
        num_channels = self.stage1_cfg['num_channels'][0]
        block_type = self.stage1_cfg['block']
        num_blocks = self.stage1_cfg['num_blocks'][0]

        block = self.blocks_dict[block_type]
        stage1_out_channels = num_channels * get_expansion(block)
        self.layer1 = self._make_layer(block, 64, stage1_out_channels,
                                       num_blocks)

        # 添加 CSPNeXtPAFPN 模块，动态指定 in_channels 和 out_channels
        self.csp_after_layer1 = CSPNeXtPAFPN(
            in_channels=[stage1_out_channels],  # 仅一个输入分支，来自 layer1
            out_channels=min(stage1_out_channels, 32),  # 输出通道：最小值保障轻量
            out_indices=(0,)  # 只有一个输入
        )
        self.fca_after_layer1 = MultiSpectralAttentionLayer(
            channel=stage1_out_channels,  # ✅ 改为适配 adapter 输出
            dct_h=56,
            dct_w=56,
            reduction=16,
            freq_sel_method='top16'
        )

        # stage 2
        self.stage2_cfg = self.extra['stage2']
        num_channels = self.stage2_cfg['num_channels']
        block_type = self.stage2_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition1 = self._make_transition_layer([stage1_out_channels],
                                                       num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.csp_stage2_branches = nn.ModuleList([
            CSPNeXtPAFPN(in_channels=[c], out_channels=c, out_indices=(0,))
            for c in num_channels
        ])
        self.fca_stage2_branches = nn.ModuleList([
            MultiSpectralAttentionLayer(
                channel=c,  # 注意：与 CSPNeXtPAFPN 的 out_channels 一致
                dct_h=28,  # 假设 stage2 输出 H,W 为 1/4 尺度（可调整）
                dct_w=28,
                reduction=16,
                freq_sel_method='top16'
            )
            for c in num_channels
        ])

        # stage 3
        self.stage3_cfg = self.extra['stage3']
        num_channels = self.stage3_cfg['num_channels']
        block_type = self.stage3_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition2 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.csp_stage3_branches = nn.ModuleList([
            CSPNeXtPAFPN(in_channels=[c], out_channels=c, out_indices=(0,))
            for c in num_channels
        ])
        self.fca_stage3_branches = nn.ModuleList([
            MultiSpectralAttentionLayer(
                channel=c,
                dct_h=14,  # 假设 stage3 分辨率为 1/8，大小大概为 14x14，可按实际调整
                dct_w=14,
                reduction=16,
                freq_sel_method='top16'
            )
            for c in num_channels
        ])

        # stage 4
        self.stage4_cfg = self.extra['stage4']
        num_channels = self.stage4_cfg['num_channels']
        block_type = self.stage4_cfg['block']

        block = self.blocks_dict[block_type]
        num_channels = [
            channel * get_expansion(block) for channel in num_channels
        ]
        self.transition3 = self._make_transition_layer(pre_stage_channels,
                                                       num_channels)

        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg,
            num_channels,
            multiscale_output=self.stage4_cfg.get('multiscale_output', False))

        self.csp_stage4_branches = nn.ModuleList([
            CSPNeXtPAFPN(in_channels=[c], out_channels=c, out_indices=(0,))
            for c in num_channels
        ])
        self.fca_stage4_branches = nn.ModuleList([
            MultiSpectralAttentionLayer(
                channel=c,
                dct_h=7,  # 假设 stage4 分辨率为 1/16（例如 7x7），根据实际特征图大小调整
                dct_w=7,
                reduction=16,
                freq_sel_method='top16'
            )
            for c in num_channels
        ])

        #调用 _freeze_stages 方法，该方法通常用于冻结某些阶段的参数（即不更新其权重）。如果 frozen_stages 设置为某个数字（例如 2），则会冻结前两个阶段的所有参数，仅更新后面的阶段。冻结某些阶段的参数可以减少计算量和内存占用，通常在预训练的情况下会使用。
        self._freeze_stages()

#构建 过渡层 (transition layer) 和访问归一化层的部分。过渡层是网络架构中用于将不同阶段的特征图通道数对齐或者调整分辨率的一种方式。
    @property
    #norm1 和 norm2 是用来访问网络中的归一化层的属性方法。
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: the normalization layer named "norm2" """
        return getattr(self, self.norm2_name)

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)#上一层的通道数，可以是一个列表，表示不同分支的通道数。
        num_branches_pre = len(num_channels_pre_layer)#当前层的通道数，同样是一个列表，表示当前层每个分支的通道数。

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)
#构建多层网络结构
    def _make_layer(self, block, in_channels, out_channels, blocks, stride=1):
        """Make layer."""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False),
                build_norm_layer(self.norm_cfg, out_channels)[1])

        layers = []
        layers.append(
            block(
                in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                with_cp=self.with_cp,
                norm_cfg=self.norm_cfg,
                conv_cfg=self.conv_cfg))
        for _ in range(1, blocks):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg))

        return nn.Sequential(*layers)
#根据给定的配置构建 HRNet 的一个阶段（stage）。每个阶段由多个模块（HRModule）组成，模块中有多个分支，每个分支执行不同尺度的特征提取。该方法的核心任务是构建每个模块，并根据模块的输出调整输入通道数
    def _make_stage(self, layer_config, in_channels, multiscale_output=True):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]

        hr_modules = []
        for i in range(num_modules):
            # multi_scale_output is only used for the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            hr_modules.append(
                HRModule(
                    num_branches,
                    block,
                    num_blocks,
                    in_channels,
                    num_channels,
                    reset_multiscale_output,
                    with_cp=self.with_cp,
                    norm_cfg=self.norm_cfg,
                    conv_cfg=self.conv_cfg,
                    upsample_cfg=self.upsample_cfg))

            in_channels = hr_modules[-1].in_channels

        return nn.Sequential(*hr_modules), in_channels
#用于冻结模型的某些层的参数
    def _freeze_stages(self):
        """Freeze parameters."""
        if self.frozen_stages >= 0:
            self.norm1.eval()
            self.norm2.eval()

            for m in [self.conv1, self.norm1, self.conv2, self.norm2]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if i == 1:
                m = getattr(self, 'layer1')
            else:
                m = getattr(self, f'stage{i}')

            m.eval()
            for param in m.parameters():
                param.requires_grad = False

            if i < 4:
                m = getattr(self, f'transition{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
#初始化 HRNet 网络中的权重
    def init_weights(self):
        """Initialize the weights in backbone."""
        super(HRNet, self).init_weights()

        if (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            # Suppress zero_init_residual if use pretrained model.
            return

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def _add_post_modules(self, in_channels, dct_h, dct_w):
        self.stage_post_csp.append(
            CSPNeXtPAFPN(
                in_channels=in_channels,
                out_channels=min(in_channels[0], 32),
                out_indices=tuple(range(len(in_channels)))
            )
        )
        self.stage_post_fca.append(
            MultiSpectralAttentionLayer(
                channel=min(in_channels[0], 32),
                dct_h=dct_h, dct_w=dct_w,
                reduction=16,
                freq_sel_method='top16'
            )
        )

    #神经网络如何通过输入数据执行前向传播的地方
    def forward(self, x):

        """Forward function."""
        print("✅ Using custom HRNet with CSPNeXtPAFPN")
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.csp_after_layer1([x])[0]  # 取出 tuple 中的第一个 tensor
        # x = self.fca_adapter(x)
        x = self.fca_after_layer1(x)

        # Stage 2
        x_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        y_list = [
            csp([y])[0] for csp, y in zip(self.csp_stage2_branches, y_list)
        ]
        y_list = [
            self.fca_stage2_branches[i](y)
            for i, y in enumerate(y_list)
        ]

        # Stage 3
        x_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)


        # 每个 stage3 分支单独经过对应的 CSP 模块
        y_list = [
            csp([y])[0] for csp, y in zip(self.csp_stage3_branches, y_list)
        ]
        y_list = [
            self.fca_stage3_branches[i](y)
            for i, y in enumerate(y_list)
        ]
        # Stage 4
        x_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)


        # 每个 stage4 分支单独经过对应的 CSPNeXtPAFPN 模块
        y_list = [
            csp([y])[0] for csp, y in zip(self.csp_stage4_branches, y_list)
        ]
        y_list = [
            self.fca_stage4_branches[i](y)
            for i, y in enumerate(y_list)
        ]
        print(f"Final output shapes: {[y.shape for y in y_list]}")
        print("✅ Using custom HRNet with CSPNeXtPAFPN")

        return tuple(y_list)

    #设置网络为训练模式，并根据配置冻结特定的层
    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
print("✅ HRNet definition loaded")

