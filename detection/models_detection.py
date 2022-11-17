import torch.nn
import torchvision.transforms
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.backbone_utils import _validate_trainable_layers, BackboneWithFPN
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.retinanet import RetinaNet
from torchvision.models.detection.ssdlite import SSD, SSDLiteHead, SSDLiteFeatureExtractorMobileNet
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelMaxPool, LastLevelP6P7
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection import _utils as det_utils
from functools import partial
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN
from typing import Callable, Any, Optional, List
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["fasterrcnn_flynet_fpn", "retinanet_flynet_fpn", "ssdlite_flynet", "keypointrcnn_flynet_fpn",
           "keypointsimple_flynet"]


def fasterrcnn_flynet_fpn(backbone, num_classes=91, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        True, trainable_backbone_layers, 6, 3)

    backbone = mobilenet_backbone(backbone, True, trainable_layers=trainable_backbone_layers)

    anchor_sizes = ((32, 64, 128, 256, 512, ), ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchors = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = FasterRCNN(backbone, num_classes, rpn_anchor_generator=anchors, **kwargs)
    return model


def retinanet_flynet_fpn(backbone, num_classes=91, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        True, trainable_backbone_layers, 6, 3)

    backbone = mobilenet_backbone(backbone, True, trainable_layers=trainable_backbone_layers,
                                  returned_layers=[3,4,5], extra_blocks=LastLevelP6P7(256, 256))

    model = RetinaNet(backbone, num_classes, **kwargs)
    return model


def ssdlite_flynet(backbone, num_classes=91, trainable_backbone_layers=None, **kwargs):
    trainable_backbone_layers = _validate_trainable_layers(
        True, trainable_backbone_layers, 6, 3)

    norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    backbone = SSDLiteFeatureExtractorMobileNet(backbone, stage_indices[-2], norm_layer, **kwargs)

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, -1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}
    model = SSD(backbone, anchor_generator, size, num_classes,
                head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer), **kwargs)
    return model


def keypointrcnn_flynet_fpn(backbone, num_classes=2, num_keypoints=17,
                              trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        num_keypoints (int): number of keypoints, default 17
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        True, trainable_backbone_layers, 6, 3)

    backbone = mobilenet_backbone(backbone, True, trainable_layers=trainable_backbone_layers)

    anchor_sizes = ((32, 64, 128, 256, 512, ), ) * 3
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchors = AnchorGenerator(anchor_sizes, aspect_ratios)

    model = KeypointRCNN(backbone, num_classes, rpn_anchor_generator=anchors,
                         num_keypoints=num_keypoints, **kwargs)
    return model


def keypointsimple_flynet(backbone, trainable_backbone_layers=None, **kwargs):
    """
    Constructs a Keypoint R-CNN model with a ResNet-50-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - keypoints (``FloatTensor[N, K, 3]``): the ``K`` keypoints location for each of the ``N`` instances, in the
          format ``[x, y, visibility]``, where ``visibility=0`` means that the keypoint is not visible.

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the keypoint loss.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - keypoints (``FloatTensor[N, K, 3]``): the locations of the predicted keypoints, in ``[x, y, v]`` format.

    For more details on the output, you may refer to :ref:`instance_seg_output`.

    Keypoint R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Example::

        >>> model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "keypoint_rcnn.onnx", opset_version = 11)

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        num_keypoints (int): number of keypoints, default 17
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
    """
    trainable_backbone_layers = _validate_trainable_layers(
        True, trainable_backbone_layers, 6, 3)

    backbone = mobilenet_backbone(backbone, False, trainable_layers=trainable_backbone_layers)
    model = SimpleBaseline(backbone)

    return model


def mobilenet_backbone(
    backbone,
    fpn,
    norm_layer=misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers=2,
    returned_layers=None,
    extra_blocks=None
):
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we wont freeze
    assert 0 <= trainable_layers <= num_stages
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    out_channels = 256
    if fpn:
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if returned_layers is None:
            returned_layers = [num_stages - 2, num_stages - 1]
        assert min(returned_layers) >= 0 and max(returned_layers) < num_stages
        return_layers = {f'{stage_indices[k]}': str(v) for v, k in enumerate(returned_layers)}

        in_channels_list = [backbone[stage_indices[i]].out_channels for i in returned_layers]
        return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)
    else:
        m = nn.Sequential(
            backbone,
            # depthwise linear combination of channels to reduce their size
            nn.Conv2d(backbone[-1].out_channels, out_channels, 1),
        )
        m.out_channels = out_channels
        return m


class SimpleBaseline(nn.Module):

    def __init__(self, backbone):
        super(SimpleBaseline, self).__init__()
        self.backbone = backbone

        # used for deconv layers
        NUM_DECONV_LAYERS = 4
        UPSAMPLE = [True, False, True, True]
        NUM_DECONV_FILTERS = [256, 256, 128, 128]
        EXPAND_SIZE = [768, 768, 768, 384]
        NUM_DECONV_KERNELS = [5, 5, 5, 5]
        FINAL_CONV_KERNEL = 1
        self.NUM_JOINTS = 17

        self.inplanes = 256
        self.deconv_with_bias = False
        self.deconv_layers = self._make_deconv_layer(
            NUM_DECONV_LAYERS,
            NUM_DECONV_FILTERS,
            NUM_DECONV_KERNELS,
            EXPAND_SIZE,
            UPSAMPLE
        )

        self.final_layer = nn.Conv2d(
            in_channels=NUM_DECONV_FILTERS[-1],
            out_channels=self.NUM_JOINTS,
            kernel_size=FINAL_CONV_KERNEL,
            stride=1,
            padding=1 if FINAL_CONV_KERNEL == 3 else 0
        )

        self.init_weights()
        self.criterion = JointsMSELoss(use_target_weight=False)

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, expand_size, upsample):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            planes = num_filters[i]
            if upsample[i]:
                layers.append(nn.Upsample(scale_factor=2, mode='bilinear'))
            layers.append(InvertedResidual(self.inplanes, planes, stride=1, expand_size=expand_size[i]))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def init_weights(self):
        for name, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if self.deconv_with_bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0)

    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        image_size = [192, 256]
        heatmap_size = [48, 64]
        sigma = 2
        target = torch.zeros([len(joints), self.NUM_JOINTS, heatmap_size[0], heatmap_size[1]]).cuda()

        # Remove "ghost persons"
        temp = []
        for image in joints:
            ghost_persons = torch.stack([torch.sum(person).floor().bool() for person in image['keypoints']])
            temp += [image['keypoints'][ghost_persons]]
        joints = temp

        # Generate gaussian
        tmp_size = sigma * 3
        size = 2 * tmp_size + 1
        xs = torch.arange(size)
        ys = torch.arange(size)
        x, y = torch.meshgrid(xs, ys)
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)).cuda()

        for image in range(len(joints)):
            for person in range(len(joints[image])):
                for joint in range(self.NUM_JOINTS):
                    mu_x = round(joints[image][person, joint, 0].item())
                    mu_y = round(joints[image][person, joint, 1].item())

                    if not mu_x and not mu_y:
                        continue

                    # Check if any part of the gaussian is in-bounds
                    l, r, b, t = int(mu_x - tmp_size), int(mu_x + tmp_size + 1), \
                           int(mu_y - tmp_size), int(mu_y + tmp_size + 1)
                    if b < heatmap_size[0] and l < heatmap_size[1] and r >= 0 and t >= 0:
                        # Usable gaussian range
                        g_y = max(0, -b), min(t, heatmap_size[0]) - b
                        g_x = max(0, -l), min(r, heatmap_size[1]) - l
                        # Image range
                        img_y = max(0, b), min(t, heatmap_size[0])
                        img_x = max(0, l), min(r, heatmap_size[1])

                        # Update target (use maximum in case of overlapping gaussians from different persons)
                        img_patch = target[image, joint, img_y[0]:img_y[1], img_x[0]:img_x[1]]
                        g_patch = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
                        target[image, joint, img_y[0]:img_y[1],
                                img_x[0]:img_x[1]] = torch.maximum(img_patch, g_patch)

        return target

    def resize_targets(self, targets, x, scale):
        for i, target in enumerate(targets):
            # Image is cxhxw, targets are [x, y, *]
            target['keypoints'][:, :, 0] *= scale[1]/x[i].size(2)
            target['keypoints'][:, :, 1] *= scale[0]/x[i].size(1)

    def get_max_preds(self, batch_heatmaps):
        # Eval is done with batch 1
        batch_size = batch_heatmaps.shape[0]
        num_joints = batch_heatmaps.shape[1]
        width = batch_heatmaps.shape[3]
        heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))

        # Multiple people (could return zero)
        # idx = (heatmaps_reshaped > 0.7).nonzero(as_tuple=True)
        # num_persons = torch.max(torch.unique(idx[1], return_counts=True)[1])      # Number of persons
        # preds = torch.zeros([batch_size, num_persons, num_joints], device=idx[0].device)
        # preds[0, :, idx[1]] = idx[2]

        # Max person
        idx = torch.argmax(heatmaps_reshaped, 2)
        maxvals = torch.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, -1))
        idx = idx.reshape((batch_size, num_joints, -1))

        preds = torch.tile(idx, (1, 1, 3)).float()

        # Targets format is [x, y]
        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = torch.floor((preds[:, :, 1]) / width).int()
        preds[:, :, 2] = 0      # Not sure what's this column for

        preds = torch.greater(maxvals, 0.0) * preds

        return preds, maxvals

    def get_final_preds(self, batch_heatmaps, scale):
        coords, maxvals = self.get_max_preds(batch_heatmaps)

        heatmap_height = batch_heatmaps.shape[2]
        heatmap_width = batch_heatmaps.shape[3]

        # Post-processing (ignore for now)
        # for i in range(coords.shape[0]):
        #     for joint in range(coords.shape[1]):
        #         hm = batch_heatmaps[i][joint]
        #         px = int(math.floor(coords[i][joint][0] + 0.5))
        #         py = int(math.floor(coords[i][joint][1] + 0.5))
        #         if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
        #             diff = np.array([hm[py][px + 1] - hm[py][px - 1],
        #                              hm[py + 1][px] - hm[py - 1][px]])
        #             coords[i][joint] += np.sign(diff) * .25

        preds = coords.cpu()

        # Transform back to original coordinates
        preds *= torch.Tensor([scale[1]/heatmap_width, scale[0]/heatmap_height, 0])
        box = torch.Tensor([torch.min(preds[:, :, 0]).item(), torch.min(preds[:, :, 1]).item(),
               torch.max(preds[:, :, 0]).item(), torch.max(preds[:, :, 1]).item()])
        # box = torch.Tensor([0, 0, 256, 192])

        return box, preds

    def forward(self, x, targets=None):
        if self.training:
            self.resize_targets(targets, x, [48, 64])
        else:
            scale = x[0].shape[1:3]
        x = torch.stack([torchvision.transforms.Resize([192, 256])(i) for i in x], dim=0)
        x = self.backbone(x)
        x = self.deconv_layers(x)
        x = torch.relu(self.final_layer(x))

        if self.training:
            losses = {}
            targets = self.generate_target(targets)
            losses.update({"mse": self.criterion(x, targets, None)})
            return losses
        else:
            boxes, keypoints = self.get_final_preds(x, scale)
            output = [{"boxes": boxes.unsqueeze(0), "scores": torch.Tensor([1]),
                       "labels": torch.Tensor([1]), "keypoints": keypoints}]
            return output


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt*10)

        return loss / num_joints


class ConvBNReLU(nn.Sequential):
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
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                      bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )
        self.out_channels = out_planes


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_size: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(expand_size))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if hidden_dim > inp:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
