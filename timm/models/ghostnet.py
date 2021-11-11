# Modified from https://github.com/openai/CLIP/blob/main/clip/model.py
# Two modifications: 
#   * account for rectangular inputs when `spatial_dim` is a tuple.
#   * `num_heads` defaults to 32 which is the setting for ResNet-50

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Tuple, Optional

__all__ = ['AttentionPool2d', 'SqueezeExcite']

class AttentionPool2d(nn.Module):
    def __init__(
        self, spacial_dim: Union[int, Tuple[int, int]], embed_dim: int,
        num_heads: int = 32, output_dim: Optional[int] = None
    ):
        super().__init__()
        h,w = spacial_dim if isinstance(spacial_dim, (list, tuple)) else [spacial_dim, spacial_dim]
        self.positional_embedding = nn.Parameter(torch.randn(h*w + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]

'''
The chunk below is take from the `timm` package. It's been copied here because
in development, we used this exact definition but at some point (I don't know when)
this definition was changed in the official repo, making the checkpoints derived
from this definition incompatible
'''

def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def sigmoid(x, inplace: bool = False):
    return x.sigmoid_() if inplace else x.sigmoid()

class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
        self.gate_fn = gate_fn

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate_fn(x_se)

## ==== TIMM

"""
An implementation of GhostNet Model as defined in:
GhostNet: More Features from Cheap Operations. https://arxiv.org/abs/1911.11907
The train script of the model is similar to that of MobileNetV3
Original model: https://github.com/huawei-noah/CV-backbones/tree/master/ghostnet_pytorch
"""
import math
from functools import partial
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import SelectAdaptivePool2d, BlurPool2d, Linear, hard_sigmoid, DropBlock2d
from timm.models.efficientnet_blocks import ConvBnAct, make_divisible
from .layers import AttentionPool2d, SqueezeExcite
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model


__all__ = ["GhostNetModified", "GhostNetArchitectures"]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": (1, 1),
        "crop_pct": 0.875,
        "interpolation": "bilinear",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "conv_stem",
        "classifier": "classifier",
        **kwargs,
    }


default_cfgs = {
    # 'ghostnet_050': _cfg(url=''),
    "ghostnet_100": _cfg(
        url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    ),
    # 'ghostnet_130': _cfg(url=''),
    "ghostnet_100_aa": _cfg(
        url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    ),
    "ghostnet_130_aa": _cfg(url=""),
    # "ghostnet_100_aa_hswish": _cfg(
    #     url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    # ),
    # "ghostnet_100_hswish": _cfg(
    #     url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    # ),

    # NOTE: The below URLs exist for now before I've experimented with these architectures
    # As I train the models and upgrade them, I should also update these URLs
    'ghostnet_100_aa_dropblock': _cfg(
        url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    ),
    'ghostnet_100_aa_attnpool_224x224': _cfg(
        url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    ),
    'ghostnet_100_aa_attnpool_224x224_dropblock': _cfg(
        url="https://github.com/huawei-noah/CV-backbones/releases/download/ghostnet_pth/ghostnet_1x.pth"
    ),
    'ghostnet_100_aa_attnpool_384x640': _cfg(
        url="https://storage.googleapis.com/cinemanet-models/IMAGENET/20211015-183412-ghostnet_100_aa_attnpool_384x640_dropblock-640/model_best.pth.tar",
        input_size = (3, 384, 640),
    ),
    'ghostnet_100_aa_attnpool_384x640_dropblock': _cfg(
        url="https://storage.googleapis.com/cinemanet-models/IMAGENET/20211015-183412-ghostnet_100_aa_attnpool_384x640_dropblock-640/model_best.pth.tar",
        input_size = (3, 384, 640),
    )
}

# This is a fucking blunder
# The args passed here were for a newer version of `timm` on the master repo
# whereas we were using an older `SqueezeExcite` definition (see `.layers.SqueezeExcite`)
# So, we ended up using the defaults -- `divisor=1, gate_fn='sigmoid'` which is suboptimal
_SE_LAYER_BOTCHED = partial(SqueezeExcite, gate_layer='hard_sigmoid', rd_round_fn=partial(make_divisible, divisor=4))

# These would be the correct arguments, if we were to train another model
_SE_LAYER = partial(SqueezeExcite, gate_fn=hard_sigmoid, divisor=4)


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1,
        act=True, act_layer=nn.ReLU,
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            act_layer(inplace=True) if act else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels, new_channels,
                kernel_size=dw_size, stride=1,
                padding=dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            act_layer(inplace=True) if act else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


class GhostBottleneck(nn.Module):
    """Ghost bottleneck w/ optional SE"""

    def __init__(
        self, in_chs, mid_chs, out_chs, dw_kernel_size=3, stride=1,
        act_layer=nn.ReLU, se_ratio=0.0, aa_layer=None, se_layer=_SE_LAYER
    ):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        use_aa = aa_layer is not None and stride == 2
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, act=True, act_layer=act_layer)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size,
                stride=1 if use_aa else stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)
            self.aa = aa_layer(mid_chs, stride=stride) if use_aa else None
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, rd_ratio=se_ratio) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, act=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs, bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)
            if self.aa is not None:
                x = self.aa(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x


from typing import Tuple

input_hw_to_fmap_size_mapping = {
    (224, 224) : (4, 4),
    (384, 384) : (6, 6),
    (640, 640) : (10, 10),
    (384, 640) : (6, 10),
}

class GhostNetModified(nn.Module):
    def __init__(
        self,
        cfgs,
        num_classes=1000,
        width=1.0,
        dropout=0.2,
        drop_block=None,
        in_chans=3,
        output_stride=32,
        aa_layer=None,
        use_attn_pooling=False,
        input_hw: Tuple[int, int] = None,
        act_layer=nn.ReLU,
        se_layer=_SE_LAYER,
    ):
        super(GhostNetModified, self).__init__()
        # setting of inverted residual blocks
        assert output_stride == 32, "only output_stride==32 is valid, dilation not supported"
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_attn_pooling = use_attn_pooling
        if use_attn_pooling:
            assert input_hw, f"`input_hw` Input (H,W) reqd. for attention pooling"
            # WARNING: we don't have all input -> fmap sizes mapped out
            attn_h, attn_w = input_hw_to_fmap_size_mapping[input_hw]
        if drop_block is not None:
            block_size = 3 # for (224, 224)
            if input_hw:
                block_size = 3 if input_hw[0] < 225 else 5
            self.drop_block = DropBlock2d(drop_block, block_size, batchwise=True)
        else:
            self.drop_block = None
        self.feature_info = []

        # building first layer
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2d(in_chans, stem_chs, 3, 2, 1, bias=False)
        self.conv_stem_aa = aa_layer(stem_chs) if aa_layer else None
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f"conv_stem"))
        self.bn1 = nn.BatchNorm2d(stem_chs)
        self.act1 = act_layer(inplace=True)
        prev_chs = stem_chs

        # building inverted residual blocks
        stages = nn.ModuleList([])
        block = GhostBottleneck
        stage_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layers.append(
                    block(
                        prev_chs, mid_chs, out_chs, k, s,
                        se_ratio=se_ratio,
                        aa_layer=aa_layer,
                        act_layer=act_layer,
                        se_layer=se_layer,
                    )
                )
                prev_chs = out_chs
            if s > 1:
                net_stride *= 2
                # if stage_idx == 7:
                #     stage_idx = 9  # HACK
                #     prev_chs = 960  # HACK
                self.feature_info.append(
                    dict(
                        num_chs=prev_chs,
                        reduction=net_stride,
                        module=f"blocks.{stage_idx}",
                    )
                )
            stages.append(nn.Sequential(*layers))
            stage_idx += 1

        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1, act_layer=act_layer)))
        self.pool_dim = prev_chs = out_chs

        self.blocks = nn.Sequential(*stages)

        # building last several layers
        self.num_features = out_chs = 1280
        
        if self.use_attn_pooling:
            self.attnpool = AttentionPool2d(
                (attn_h, attn_w), embed_dim=prev_chs, num_heads=32, output_dim=out_chs
            )
        else:
            self.global_pool = SelectAdaptivePool2d(pool_type="avg")
            self.conv_head = nn.Conv2d(prev_chs, out_chs, 1, 1, 0, bias=True)
        self.act2 = act_layer(inplace=True)
        self.classifier = Linear(out_chs, num_classes)

        last_ft_info = self.feature_info[-1]
        last_ft_info["num_chs"] = 960
        last_ft_info["module"] = "blocks.9"

    def get_classifier(self):
        return self.classifier

    def reset_classifier(self, num_classes, global_pool="avg"):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.classifier = Linear(self.pool_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        if self.conv_stem_aa is not None:
            x = self.conv_stem_aa(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        #print(f"Feat shape: {x.shape}")

        # If using drop block, we apply it now to the last feature map only
        if self.drop_block is not None:
            x = self.drop_block(x)

        # Use either average pooling or Attention Pooling 2D
        if self.use_attn_pooling:
            x = self.attnpool(x)
        else:
            x = self.global_pool(x)
            x = self.conv_head(x)
        #print(f"Post Feat Conv shape: {x.shape}")
        x = self.act2(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if not self.use_attn_pooling:
            if not self.global_pool.is_identity():
                x = x.view(x.size(0), -1)
        if self.dropout > 0.0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        return x

    def load_huawei_pretrained_model(self):
        # `timm`'s weight loading is strict. If we're using attention pooling,
        # then we need to load weight in unstrict mode
        print(f"{'-' * 40} Loading PreTrained Model {'-' * 40}")
        sd = torch.hub.load_state_dict_from_url(default_cfgs['ghostnet_100']['url'], map_location='cpu')
        self.load_state_dict(sd, strict=False)

    def load_custom_weights_384x640(self):
        print(f"{'-' * 30} Loading Custom PreTrained Model {'-' * 30}")
        sd = torch.hub.load_state_dict_from_url(default_cfgs['ghostnet_100_aa_attnpool_384x640']['url'], map_location='cpu')
        sd = sd['state_dict_ema']
        self.load_state_dict(sd, strict=True)

def _create_ghostnet(variant, width=1.0, pretrained=False, **kwargs):
    """
    Constructs a GhostNetModified model
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[5, 72, 40, 0.25, 2]],
        [[5, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 0.25, 1],
            [3, 672, 112, 0.25, 1],
        ],
        # stage5
        [[5, 672, 160, 0.25, 2]],
        [
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 0.25, 1],
        ],
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        GhostNetModified,
        variant,
        pretrained,
        default_cfg=default_cfgs[variant],
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )


# # NOTE: Replace 'ghostnet_100
# @register_model
# def ghostnet_100(pretrained=True, **kwargs):
#     model = _create_ghostnet(
#         "ghostnet_100", width=1.0, pretrained=pretrained, aa_layer=BlurPool2d, **kwargs
#     )
#     return model


@register_model
def ghostnet_100_aa(pretrained=True, **kwargs):
    model = _create_ghostnet(
        "ghostnet_100_aa", width=1.0, pretrained=pretrained, aa_layer=BlurPool2d, **kwargs
    )
    return model

@register_model
def ghostnet_130_aa(pretrained=False, **kwargs):
    """ GhostNet-1.3x Anti Aliased """
    model = _create_ghostnet(
        'ghostnet_130_aa', width=1.3, pretrained=pretrained, aa_layer=BlurPool2d, **kwargs
    )
    return model


# NOTE: Discarding hswish models as the improvement is marginal, but runtime penalty
# is not

# @register_model
# def ghostnet_100_aa_hswish(pretrained=True, **kwargs):
#     model = _create_ghostnet(
#         "ghostnet_100",
#         width=1.0,
#         pretrained=pretrained,
#         aa_layer=BlurPool2d,
#         act_layer=nn.Hardswish,
#         **kwargs,
#     )
#     return model


# @register_model
# def ghostnet_100_hswish(pretrained=True, **kwargs):
#     model = _create_ghostnet(
#         "ghostnet_100", width=1.0, pretrained=pretrained, act_layer=nn.Hardswish
#     )
#     return model


# @register_model
# def ghostnet_050(pretrained=False, **kwargs):
#     """GhostNetModified-0.5x"""
#     model = _create_ghostnet("ghostnet_050", width=0.5, pretrained=pretrained, **kwargs)
#     return model


# @register_model
# def ghostnet_130(pretrained=False, **kwargs):
#     """GhostNetModified-1.3x"""
#     model = _create_ghostnet("ghostnet_130", width=1.3, pretrained=pretrained, **kwargs)
#     return model

@register_model
def ghostnet_100_aa_dropblock(pretrained=True, drop_block=0.1, **kwargs):
    m: GhostNetModified = _create_ghostnet(
        "ghostnet_100_aa_dropblock", width=1.0, pretrained=pretrained, aa_layer=BlurPool2d,
        drop_block=drop_block, **kwargs
    )
    return m


@register_model
def ghostnet_100_aa_attnpool_224x224(pretrained=True, **kwargs):
    m: GhostNetModified = _create_ghostnet(
        "ghostnet_100_aa_attnpool_224x224", width=1.0, pretrained=False, aa_layer=BlurPool2d,
        use_attn_pooling=True, input_hw=(224,224)
    )
    if pretrained:
        m.load_huawei_pretrained_model()
    return m


@register_model
def ghostnet_100_aa_attnpool_224x224_dropblock(pretrained=True, drop_block=0.1, **kwargs):
    m: GhostNetModified = _create_ghostnet(
        "ghostnet_100_aa_attnpool_224x224_dropblock", width=1.0, pretrained=False, aa_layer=BlurPool2d,
        use_attn_pooling=True, input_hw=(224,224), drop_block=drop_block,
    )
    if pretrained:
        m.load_huawei_pretrained_model()
    return m


@register_model
def ghostnet_100_aa_attnpool_384x640(pretrained=True, **kwargs):
    m: GhostNetModified = _create_ghostnet(
        "ghostnet_100_aa_attnpool_384x640", width=1.0, pretrained=False, aa_layer=BlurPool2d,
        use_attn_pooling=True, input_hw=(384,640),
        se_layer=_SE_LAYER_BOTCHED  # For compatibility, remove if new model's trained
    )
    if pretrained:
        m.load_custom_weights_384x640()
    return m


@register_model
def ghostnet_100_aa_attnpool_384x640_dropblock(pretrained=True, drop_block=0.1, **kwargs):
    m: GhostNetModified = _create_ghostnet(
        "ghostnet_100_aa_attnpool_384x640_dropblock", width=1.0, pretrained=False, aa_layer=BlurPool2d,
        use_attn_pooling=True, input_hw=(384,640), drop_block=drop_block,
        se_layer=_SE_LAYER_BOTCHED  # For compatibility, remove if new model's trained
    )
    if pretrained:
        m.load_custom_weights_384x640()
    return m

class GhostNetArchitectures:
    pass

arch_options = []
for k in default_cfgs.keys():
    if not k == "ghostnet_100":
        arch_options.append(k)
        setattr(GhostNetArchitectures, k, eval(k))

GhostNetArchitectures.OPTIONS = arch_options
