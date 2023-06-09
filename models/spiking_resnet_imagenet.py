import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ['SpikingNFResNet', 'spiking_nfresnet34', 'spiking_nfresnet50', 'spiking_nfresnet101']


class ScaledWSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gain=True, eps=1e-4):
        super(ScaledWSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x):
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


def wsconv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return ScaledWSConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=True, gain=True, dilation=dilation)


def wsconv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return ScaledWSConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True, gain=True)


#TODO: stochastic depth
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, beta=1.0, alpha=1.0,
                 neuron: callable = None, **kwargs):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = wsconv3x3(inplanes, planes, stride)
        self.conv2 = wsconv3x3(planes, planes)

        self.sn1 = neuron(**kwargs)
        self.sn2 = neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        out = x / self.beta

        out = self.sn1(out)
        out = out * 2.74

        if self.downsample is not None:
            identity = self.downsample(out)
        else:
            identity = x

        out = self.conv1(out)
        out = self.sn2(out)
        out = out * 2.74

        out = self.conv2(out)

        out = out * self.skipinit_gain * self.alpha + identity

        return out



#TODO: stochastic depth
class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, beta=1.0, alpha=1.0,
                 neuron: callable = None,  **kwargs):
        super(Bottleneck, self).__init__()


        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = wsconv1x1(inplanes, width)
        self.conv2 = wsconv3x3(width, width, stride, groups, dilation)
        self.conv3 = wsconv1x1(width, planes * self.expansion)


        self.sn1 = neuron(**kwargs)
        self.sn2 = neuron(**kwargs)
        self.sn3 = neuron(**kwargs)
        self.downsample = downsample
        self.stride = stride

        self.beta, self.alpha = beta, alpha
        self.skipinit_gain = nn.Parameter(torch.zeros(()))


    def forward(self, x):
        out = x / self.beta

        out = self.sn1(out)
        out = out * 2.74

        if self.downsample is not None:
            identity = self.downsample(out)
        else:
            identity = x


        out = self.conv1(out)
        out = self.sn2(out)
        out = out * 2.74

        out = self.conv2(out)
        out = self.sn3(out)
        out = out * 2.74

        out = self.conv3(out)
        out = out * self.skipinit_gain * self.alpha + identity

        return out



class SpikingNFResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, 
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, neuron: callable = None,
                 alpha=0.2, neuron_dropout=0.0, **kwargs):
        super(SpikingNFResNet, self).__init__()
        self.alpha = alpha
        self.drop_rate = neuron_dropout

        self.inplanes = 64
        self.dilation = 1
        self.c_in = kwargs.get('c_in', 3)
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = ScaledWSConv2d(self.c_in, self.inplanes, kernel_size=7, stride=2, padding=3, bias=True, gain=True)
        self.maxpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        expected_var = 1.0
        self.layer1, expected_var = self._make_layer(block, 64, layers[0], alpha=self.alpha, var=expected_var, neuron=neuron, **kwargs)
        self.layer2, expected_var = self._make_layer(block, 128, layers[1], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[0], neuron=neuron, **kwargs)
        self.layer3, expected_var = self._make_layer(block, 256, layers[2], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[1], neuron=neuron, **kwargs)
        self.layer4, expected_var = self._make_layer(block, 512, layers[3], stride=2, alpha=self.alpha, var=expected_var,
                                                     dilate=replace_stride_with_dilation[2], neuron=neuron, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        torch.nn.init.zeros_(self.fc.weight)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, ScaledWSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1, alpha=1.0, var=1.0, dilate=False, neuron: callable = None, **kwargs):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = wsconv1x1(self.inplanes, planes * block.expansion, stride)

        layers = []
        beta = var ** 0.5
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, beta, alpha, neuron, **kwargs))
        self.inplanes = planes * block.expansion
        if downsample != None:
            var = 1. + self.alpha ** 2
        else:
            var += self.alpha ** 2
        for _ in range(1, blocks):
            beta = var ** 0.5
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, beta=beta, alpha=alpha, neuron=neuron, **kwargs))
            var += self.alpha ** 2

        return nn.Sequential(*layers), var

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)



def _spiking_resnet(arch, block, layers, neuron, **kwargs):
    model = SpikingNFResNet(block, layers, neuron=neuron, **kwargs)

    return model


def spiking_nfresnet34(neuron: callable=None, **kwargs):
    return _spiking_resnet('resnet34', BasicBlock, [3, 4, 6, 3], neuron, **kwargs)


def spiking_nfresnet50(neuron: callable=None, **kwargs):
    return _spiking_resnet('resnet50', Bottleneck, [3, 4, 6, 3], neuron, **kwargs)


def spiking_nfresnet101(neuron: callable=None, **kwargs):
    return _spiking_resnet('resnet101', Bottleneck, [3, 4, 23, 3], neuron, **kwargs)
