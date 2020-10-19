import torch
import torch.nn as nn
import torchvision.models.resnet


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualStage(nn.Module):
    def __init__(self, in_chs, out_chs, stride, depth, block_fn=BasicBlock):
        super(ResidualStage, self).__init__()
        self.blocks = nn.Sequential()
        downsample = nn.Sequential(conv1x1(in_chs, out_chs, stride), nn.BatchNorm2d(out_chs)) if stride != 1 else None
        self.blocks.add_module('0', block_fn(in_chs, out_chs, stride, downsample))
        for i in range(1, depth):
            self.blocks.add_module(str(i), block_fn(out_chs, out_chs, 1))

    def forward(self, x):
        y = self.blocks(x)
        return y


class ResNet(nn.Module):
    def __init__(self, cfg, fpn=False):
        super(ResNet, self).__init__()
        self.fpn = fpn
        self.stem = nn.Sequential(nn.Conv2d(3, bias=False, **cfg['stem']),
                                  nn.BatchNorm2d(cfg['stem']['out_channels']),
                                  nn.ReLU(inplace=True))
        self.stage1 = ResidualStage(**cfg['stage1'])
        self.stages = nn.ModuleList([ResidualStage(**cfg['stage2']),
                                     ResidualStage(**cfg['stage3']),
                                     ResidualStage(**cfg['stage4'])])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        outputs = [x]
        for module in self.stages:
            outputs.append(module(outputs[-1]))

        return outputs[1:]


if __name__ == '__main__':
    # net = ResidualStage(64, 64, 2, 2)
    net = ResNet()
    dummy = torch.rand((1, 3, 112, 112))
    output = net(dummy)
    for out in output:
        print(out.size())
    # print(net)
