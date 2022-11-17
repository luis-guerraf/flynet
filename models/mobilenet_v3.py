import torch

from functools import partial
from torch import nn, Tensor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from timm.models.layers import DropPath

from .flynet_utils import _make_divisible, ConvBNActivation, eca_layer, eca_softmax, channelwise_max, Quantize

k_depthwise = None

__all__ = ["MobileNetV3", "mobilenet_v3_large", "mobilenet_v3_small"]


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidualConfig:

    def __init__(self, input_channels: int, kernel: int, expanded_channels: int, out_channels: int, use_se: bool,
                 activation: str, stride: int, dilation: int, width_mult: float):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):

    def __init__(self, cnf: InvertedResidualConfig, norm_layer: Callable[..., nn.Module],
                 drop_path=0.0, high_dim_res=None):
        super().__init__()

        self.k = k_depthwise
        self.stride = cnf.stride
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.eca_residual_low_dim = eca_layer()
        self.eca_residual_high_dim = eca_layer()

        if not (1 <= cnf.stride <= 2):
            raise ValueError('illegal stride value')

        # Low dimensional residual
        if cnf.input_channels == cnf.out_channels and self.stride != 1:
            self.downsample_ch_low_dim = ConvBNActivation(cnf.input_channels, cnf.input_channels,
                                                  kernel_size=3, stride=self.stride,
                                                  groups=cnf.input_channels, norm_layer=norm_layer,
                                                  activation_layer=nn.Identity)
        if cnf.input_channels < cnf.out_channels:
            if self.stride != 1:
                self.downsample_ch_1_low_dim = ConvBNActivation(cnf.input_channels, cnf.input_channels,
                                                      kernel_size=3, stride=self.stride,
                                                      groups=cnf.input_channels, norm_layer=norm_layer,
                                                      activation_layer=nn.Identity)
            channels_dif = cnf.out_channels - cnf.input_channels
            self.downsample_ch_2_low_dim = ConvBNActivation(channels_dif, channels_dif,
                                                      kernel_size=3, stride=self.stride,
                                                      groups=channels_dif, norm_layer=norm_layer,
                                                      activation_layer=nn.Identity)

        # High dimensional residual
        if high_dim_res is not None:
            if high_dim_res == cnf.expanded_channels and self.stride != 1:
                self.downsample_ch_high_dim = ConvBNActivation(high_dim_res, high_dim_res,
                                                      kernel_size=3, stride=self.stride,
                                                      groups=high_dim_res, norm_layer=norm_layer,
                                                      activation_layer=nn.Identity)
            if high_dim_res < cnf.expanded_channels:
                if self.stride != 1:
                    self.downsample_ch_1_high_dim = ConvBNActivation(high_dim_res, high_dim_res,
                                                          kernel_size=3, stride=self.stride,
                                                          groups=high_dim_res, norm_layer=norm_layer,
                                                          activation_layer=nn.Identity)
                channels_dif = cnf.expanded_channels - high_dim_res
                if channels_dif <= high_dim_res:
                    self.downsample_ch_2_high_dim = ConvBNActivation(channels_dif, channels_dif,
                                                          kernel_size=3, stride=self.stride,
                                                          groups=channels_dif, norm_layer=norm_layer,
                                                          activation_layer=nn.Identity)
                else:
                    channels_dif = _make_divisible(channels_dif, high_dim_res)
                    self.downsample_ch_2_high_dim = ConvBNActivation(high_dim_res, channels_dif,
                                                          kernel_size=3, stride=self.stride,
                                                          groups=high_dim_res, norm_layer=norm_layer,
                                                          activation_layer=nn.Identity)

        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # Pointwise
        if cnf.expanded_channels != cnf.input_channels:
            self.pointwise = nn.Sequential(ConvBNActivation(cnf.input_channels, cnf.expanded_channels,
                                    kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer),
                                    eca_layer())
        else:
            self.pointwise = nn.Identity()

        # Depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        self.depthwise = nn.Sequential(ConvBNActivation(cnf.expanded_channels, cnf.expanded_channels * self.k,
                                       kernel_size=cnf.kernel, stride=stride, groups=cnf.expanded_channels,
                                       norm_layer=norm_layer, activation_layer=nn.Identity),
                        eca_softmax(k=self.k),
                        channelwise_max(k=self.k),
                        norm_layer(cnf.expanded_channels),
                        eca_layer())

        # Project
        self.project = nn.Sequential(nn.Conv2d(cnf.expanded_channels, cnf.out_channels, 1, 1, 0, bias=False),
                                     norm_layer(cnf.out_channels),
                                     eca_layer())

        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def match_low_dim_residual(self, input, channels):
        if input.shape[1] > channels:
            # First layer
            input = input[:, 0:channels]
        if input.shape[1] == channels and self.stride != 1:
            input = self.downsample_ch_low_dim(input)
        if input.shape[1] < channels:
            dif = channels - input.shape[1]
            if self.stride == 1:
                input = torch.cat([input, self.downsample_ch_2_low_dim(input[:, 0:dif])], dim=1)
            else:
                input = torch.cat([self.downsample_ch_1_low_dim(input),
                                   self.downsample_ch_2_low_dim(input[:, 0:dif])], dim=1)

        input = self.eca_residual_low_dim(input)
        return input

    def match_high_dim_residual(self, input, channels):
        if input is None:
            return 0
        if input.shape[1] > channels:
            # First layer
            input = input[:, 0:channels]
        if input.shape[1] == channels and self.stride != 1:
            input = self.downsample_ch_high_dim(input)
        if input.shape[1] < channels:
            dif = channels - input.shape[1]
            if self.stride == 1:
                input = torch.cat([input, self.downsample_ch_2_high_dim(input[:, 0:dif])], dim=1)[:, :channels]
            else:
                input = torch.cat([self.downsample_ch_1_high_dim(input),
                                   self.downsample_ch_2_high_dim(input[:, 0:dif])], dim=1)[:, :channels]

        input = self.eca_residual_high_dim(input)
        return input

    def forward(self, input) -> Tensor:
        if isinstance(input, Tuple):
            input, high_dim_res = input
        else:
            high_dim_res = None
        result = self.pointwise(input)
        high_dim_res = self.depthwise(result) + self.match_high_dim_residual(high_dim_res, result.shape[1])
        result = self.project(high_dim_res)

        # Drop path must be in the form drop_path(x) + x
        result = self.drop_path(result) + self.match_low_dim_residual(input, result.shape[1])
        return (result, high_dim_res)


class MobileNetV3(nn.Module):

    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        last_channel: int,
        num_classes: int = 1000,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            last_channel (int): The number of channels on the penultimate layer
            num_classes (int): Number of classes
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            dropout (float): The droupout probability
        """
        super().__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(ConvBNActivation(3, firstconv_output_channels, kernel_size=3, stride=2, norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))

        # building inverted residual blocks
        for i, cnf in enumerate(inverted_residual_setting):
            block_dpr = kwargs['drop_path'] * i / (len(inverted_residual_setting)-1)
            if i > 0:
                layers.append(block(cnf, norm_layer, block_dpr,
                                    high_dim_res=inverted_residual_setting[i-1].expanded_channels))
            else:
                layers.append(block(cnf, norm_layer, block_dpr))

        # building last several layers
        lastconv_input_channels = inverted_residual_setting[-1].out_channels
        lastconv_output_channels = 6 * lastconv_input_channels
        self.last_conv = ConvBNActivation(lastconv_input_channels, lastconv_output_channels, kernel_size=1,
                                       norm_layer=norm_layer, activation_layer=nn.Hardswish)

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(lastconv_output_channels, last_channel),
            nn.Hardswish(inplace=True),
            Quantize(),
            nn.Linear(last_channel, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.last_conv(x[0])

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # C1
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # C2
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # C3
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1280 // reduce_divider)  # C5
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # C1
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # C2
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # C3
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2, dilation),  # C4
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1, dilation),
        ]
        last_channel = adjust_channels(1024 // reduce_divider)  # C5
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, last_channel


def _mobilenet_v3(
    arch: str,
    inverted_residual_setting: List[InvertedResidualConfig],
    last_channel: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
):
    model = MobileNetV3(inverted_residual_setting, last_channel, **kwargs)
    return model


def mobilenet_v3_large(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a large MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_large"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)


def mobilenet_v3_small(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV3:
    """
    Constructs a small MobileNetV3 architecture from
    `"Searching for MobileNetV3" <https://arxiv.org/abs/1905.02244>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    arch = "mobilenet_v3_small"
    inverted_residual_setting, last_channel = _mobilenet_v3_conf(arch, **kwargs)
    return _mobilenet_v3(arch, inverted_residual_setting, last_channel, pretrained, progress, **kwargs)
