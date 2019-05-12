import torch
import torch.nn as nn


def _binarize_weight(weight):
    alpha = weight.abs().mean((1, 2, 3), keepdim=True)
    return weight.sign_().mul_(alpha)


class BinaryWeights(nn.Module):
    def __init__(self, wrapped, exclude_layers=()):
        super().__init__()
        self.wrapped = wrapped
        self.conv_weights = [m.weight for m in wrapped.modules()
                             if isinstance(m, nn.Conv2d)]
        if exclude_layers:
            exclude_layers = {i if i >= 0 else i + len(self.conv_weights)
                              for i in exclude_layers}
            self.conv_weights = [w for i, w in enumerate(self.conv_weights)
                                 if i not in exclude_layers]
        self.full_weights = [w.detach().clone() for w in self.conv_weights]

    def forward(self, x):
        return self.wrapped(x)

    def binarize_weights(self):
        for weight, full_weight in zip(self.conv_weights, self.full_weights):
            full_weight.copy_(weight)
            _binarize_weight(weight.data)

    def restore_weights(self):
        for weight, full_weight in zip(self.conv_weights, self.full_weights):
            weight.data.copy_(full_weight)
