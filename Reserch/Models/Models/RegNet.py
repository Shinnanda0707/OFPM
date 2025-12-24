import math
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torchvision.ops.misc import SqueezeExcitation
from torchvision.models._utils import _make_divisible

__all__ = ["RegNet"]


# Conv Norm Activation
class ConvNormActivation(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm3d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: int = 1,
        inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [
            torch.nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                dilation=dilation,
                groups=groups,
                bias=norm_layer is None,
            )
        ]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels


# Stem
class SimpleStemIN(ConvNormActivation):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
    ) -> None:
        super().__init__(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=2,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )


# Bottleneck
class BottleneckTransform(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(out_planes * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = ConvNormActivation(
            in_planes,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        layers["b"] = ConvNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            groups=g,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if se_ratio:
            width_se_out = int(round(se_ratio * in_planes))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = ConvNormActivation(
            w_b,
            out_planes,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: Any,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.proj = None
        should_proj = (in_planes != out_planes) or (stride != 1)
        if should_proj:
            self.proj = ConvNormActivation(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
            )

        self.f = BottleneckTransform(
            in_planes,
            out_planes,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )

        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "BlockParams":
        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")

        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))
        block_widths = (
            (
                torch.round(torch.divide(w_0 * torch.pow(w_m, block_capacity), QUANT))
                * QUANT
            )
            .int()
            .tolist()
        )
        num_stages = len(set(block_widths))

        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = (
            torch.diff(torch.tensor([d for d, t in enumerate(splits) if t]))
            .int()
            .tolist()
        )

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [bottleneck_multiplier] * num_stages
        group_widths = [group_width] * num_stages

        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        ws_bot = [
            _make_divisible(w_bot, g) for w_bot, g in zip(widths, group_widths_min)
        ]
        stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 2,
        stem_width: int = 64,
        stem_type: Optional[Callable[..., nn.Module]] = None,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if stem_type is None:
            stem_type = SimpleStemIN
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        self.stem = stem_type(3, stem_width, norm_layer, activation,)

        current_width = stem_width

        blocks = []
        for (
            i,
            (width_out, stride, depth, group_width, bottleneck_multiplier,),
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        print(x.shape)
        x = self.stem(x)
        print(x.shape)
        x = self.trunk_output(x)
        print(x.shape)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


def _regnet(
    arch: str,
    block_params: BlockParams,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> RegNet:
    norm_layer = kwargs.pop(
        "norm_layer", partial(nn.BatchNorm3d, eps=1e-05, momentum=0.1)
    )
    model = RegNet(block_params, norm_layer=norm_layer, **kwargs)
    return model


def regnet_y_400mf(
    pretrained: bool = False, progress: bool = True, **kwargs: Any
) -> RegNet:
    params = BlockParams.from_init_params(
        depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **kwargs
    )
    return _regnet("regnet_y_400mf", params, pretrained, progress, **kwargs)
