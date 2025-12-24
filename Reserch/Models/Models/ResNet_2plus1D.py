from audioop import bias
from typing import Any, Callable, List, Optional, Type, Union
from numpy import identity
import torch
from torch import Tensor
import torch.nn as nn


def conv(
    in_planes: int,
    out_planes: int,
    midplanes: Optional[int] = None,
    stride: int = 1,
    padding: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            in_planes,
            midplanes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False,
        ),
        nn.BatchNorm3d(midplanes),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            midplanes,
            out_planes,
            kernel_size=(3, 1, 1),
            stride=(stride, 1, 1),
            padding=(padding, 0, 0),
            bias=False,
        ),
    )


def stem() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(
            3,
            45,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        ),
        nn.BatchNorm3d(45),
        nn.ReLU(inplace=True),
        nn.Conv3d(
            45,
            64,
            kernel_size=(3, 1, 1),
            stride=(1, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        ),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        midplanes = midplanes = (in_planes * planes * 3 * 3 * 3) // (
            in_planes * 3 * 3 + 3 * planes
        )

        self.conv1 = nn.Sequential(
            conv(in_planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            conv(planes, planes, midplanes), nn.BatchNorm3d(planes),
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        midplanes = (in_planes * planes * 3 * 3 * 3) // (in_planes * 3 * 3 + 3 * planes)

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            conv(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion),
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 2,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()

        self.in_planes = 64
        self.stem = stem()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc1 = nn.Linear(512 * block.expansion, 512 * block.expansion // 2)
        self.fc2 = nn.Linear(512 * block.expansion // 2, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = (stride, stride, stride)
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=ds_stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv))

        return nn.Sequential(*layers)


def resnet18_2_1d() -> ResNet:
    return ResNet(block=BasicBlock, layers=[2, 2, 2, 2])
