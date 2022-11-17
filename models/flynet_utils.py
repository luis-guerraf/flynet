import torch
from torch import nn
from torch import Tensor
from typing import Callable, Any, Optional, List
import torch.nn.functional as F


class Quantize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # Activation quantization
        input = self.DoReFa(input, self.bitA, mode='adaptive')
        return input

    def DoReFa(self, x, numBits, mode=None):
        # Assumed symmetric distribution of weights (i.e. range [-val, val])

        # Bring to range [0, 1] reducing impact of large values
        # if torch.min(x) >= 0:
        if mode == 'adaptive':
            temp1 = torch.min(x)
            w_q = x - temp1
            temp2 = torch.max(w_q)
            w_q = w_q / temp2
        else:
            w_q = torch.clamp(x, min=-1.0, max=1.0).div(2) + 0.5

        # Quantize to k bits in range [0, 1]
        w_q = self.quantize(w_q, numBits)

        # Affine to bring back to original range
        if mode == 'adaptive':
            w_q = w_q * temp2
            w_q = w_q + temp1
        else:
            w_q *= 2
            w_q -= 1

        return w_q

    def quantize(self, x, k):
        n = float(2**k - 1.0)
        x = self.RoundNoGradient.apply(x, n)
        return x

    class RoundNoGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, n):
            return torch.round(x*n)/n

        @staticmethod
        def backward(ctx, g):
            return g, None

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=7):
        super(eca_layer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(2, 4, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        var, mean = torch.var_mean(x, dim=[-1, -2], unbiased=False, keepdim=True)
        y = torch.cat([var, mean], dim=-2)
        # y = self.avg_pool(x)

        # Scalings
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Dynamic Relu
        y = self.sigmoid(y)
        y1, y2, y3, y4 = y.split(1, dim=2)

        x1 = x * y1 + y2
        x2 = x * y3 + y4

        x = torch.maximum(x1, x2)

        return x

class eca_softmax(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k, k_size=7):
        super(eca_softmax, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.DyRelu = 2
        self.conv = nn.Conv1d(2, 1*self.DyRelu, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.k = k
        self.T = 10

    def forward(self, x):
        B, C, H, W = x.shape

        # feature descriptor on the global spatial information
        var, mean = torch.var_mean(x, dim=[-1, -2], unbiased=False, keepdim=True)
        y = torch.cat([var, mean], dim=-2)
        # y = self.avg_pool(x)

        # Scalings
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).view(B, -1, self.k * self.DyRelu)

        y = (y/self.T).softmax(dim=-1)
        y = y.view(B, -1, self.k, self.DyRelu).sum(dim=-1).view(B, C, 1, 1)

        return x * y

class channelwise_max(nn.Module):
    def __init__(self, k) -> None:
        super(channelwise_max, self).__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        x = x.view(B, -1, self.k, H, W)
        x = x.max(dim=2).values
        return x

class channelwise_sum_relu(nn.Module):
    def __init__(self, k) -> None:
        super(channelwise_sum_relu, self).__init__()
        self.k = k

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.size()
        x = x.view(B, -1, self.k, H, W)
        x = F.relu(x.sum(dim=2))
        return x

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        dilation: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2 * dilation
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        super(ConvBNActivation, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes

# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation
