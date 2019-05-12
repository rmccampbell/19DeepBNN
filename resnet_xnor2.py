import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNetXNOR', 'resnet18_xnor', 'resnet34_xnor', 'resnet50_xnor',
           'resnet101_xnor', 'resnet152_xnor']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class BinActivSign(torch.autograd.Function):
    """Sign component of binary activation function from XNOR-Net

    Computes sign in the forward pass, and approximates the gradient
    using the "straight-through estimator" 1[|x| <= 1] for the backward pass.

    See https://arxiv.org/pdf/1603.05279.pdf,
        https://arxiv.org/pdf/1602.02830.pdf
    """

    @staticmethod
    def forward(ctx, input):
        """sign(input)"""
        ctx.save_for_backward(input)
##        output = torch.ones_like(input)
##        output[input < 0] = -1
##        return output
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output):
        """Straight-through estimator"""
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[abs(input) > 1] = 0
        return grad_input


class BinActivConv(nn.Module):
    """Binary activation function + binary conv from XNOR-Net
    See https://arxiv.org/pdf/1603.05279.pdf
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size,
                              stride=stride, padding=padding, bias=False)
        kh, kw = self.conv.kernel_size
        avg_kernel = None
        if kh * kw > 1 or stride != 1:
            avg_kernel = torch.full((1, 1, kh, kw), 1/(kh*kw))
        self.register_buffer('avg_kernel', avg_kernel)

    def forward(self, input):
        sign = BinActivSign.apply(input)
        A = K = input.abs().mean(1, keepdim=True)
        if self.avg_kernel is not None:
            K = F.conv2d(A, self.avg_kernel, 
                         stride=self.conv.stride, padding=self.conv.padding)
        return self.conv(sign) * K


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = BinActivConv(inplanes, planes, 3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = BinActivConv(planes, planes, 3, padding=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = BinActivConv(inplanes, planes, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = BinActivConv(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = BinActivConv(planes, planes * 4, kernel_size=1)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.bn1(x)
        out = self.conv1(x)

        out = self.bn2(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out += residual

        return out


class ResNetXNOR(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNetXNOR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                BinActivConv(self.inplanes, planes * block.expansion,
                             kernel_size=1, stride=stride),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _load_pretrained(model, state_dict):
    orig_state = model.state_dict()
    for name, param in state_dict.items():
        if 'bn' in name or 'downsample.1' in name:
            continue
        if name.startswith('layer') and ('conv' in name or 'downsample.0' in name):
            name = name[:-len('.weight')] + '.conv.weight'
        orig_state[name].copy_(param)


def resnet18_xnor(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetXNOR(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34_xnor(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetXNOR(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50_xnor(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetXNOR(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101_xnor(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetXNOR(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152_xnor(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetXNOR(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_pretrained(model, model_zoo.load_url(model_urls['resnet152']))
    return model
