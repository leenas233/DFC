import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "../")  # run under the current directory
from common.network import *

mode_pad_dict = {"s": 1, "d": 2, "y": 2, "e": 3, "h": 3, "o": 3}


def round_func(input):
    # Backward Pass Differentiable Approximation (BPDA)
    # This is equivalent to replacing round function (non-differentiable)
    # with an identity function (differentiable) only when backward,
    forward_value = torch.round(input)
    out = input.clone()
    out.data = forward_value.data
    return out


class SRNets_Origin(nn.Module):
    """ A MuLUT network with Residual connection"""

    def __init__(self, nf=64, scale=4, modes=['s', 'd', 'y'], stages=2):
        super(SRNets_Origin, self).__init__()
        self.modes = modes
        self.stages = stages

        module_dict = dict()

        for s in range(stages):  # 2-stage
            if (s + 1) == stages:
                upscale = scale
                flag = "N"
            else:
                upscale = None
                flag = "1"
            for mode in modes:
                module_dict["s{}_{}".format(str(s + 1), mode)] = SRNet("{}x{}".format(mode.upper(), flag), nf=nf,
                                                                       upscale=upscale)
        self.module_dict = nn.ModuleDict(module_dict)

    def forward(self, x, phase='train'):
        modes, stages = self.modes, self.stages
        # Stage 1
        for s in range(stages):
            pred = 0
            for mode in modes:
                sub_module = self.module_dict["s{}_{}".format(str(s + 1), mode)]

                pad = mode_pad_dict[mode]
                for r in [0, 1, 2, 3]:
                    pred += round_func(
                        torch.rot90(sub_module(F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')),
                                    (4 - r) % 4, [2, 3]) * 127)
            if s + 1 == stages:
                avg_factor, bias, norm = len(modes), 0, 1
                x = round_func((pred / avg_factor) + bias)
                if phase == "train":
                    x = x / 255.0
            else:
                avg_factor, bias, norm = len(modes) * 4, 127, 255.0
                x = round_func(torch.clamp((pred / avg_factor) + bias, 0, 255)) / norm
        return x


class SRNets_Res(nn.Module):
    """ A MuLUT network with Residual connection"""

    def __init__(self, nf=64, scale=4, modes=['s', 'd', 'y'], stages=2):
        super(SRNets_Res, self).__init__()
        self.modes = modes
        self.upscale = scale
        self.stages = stages

        module_dict = dict()

        for s in range(stages):  # 2-stage
            if (s + 1) == stages:
                upscale = scale
                flag = "N"
            else:
                upscale = None
                flag = "1"
            for mode in modes:
                module_dict["s{}_{}".format(str(s + 1), mode)] = SRNet("{}x{}".format(mode.upper(), flag), nf=nf,
                                                                       upscale=upscale)
        self.module_dict = nn.ModuleDict(module_dict)
        self.pixel_shuffle = nn.PixelShuffle(scale)

    def forward(self, x, phase='train'):
        modes, stages = self.modes, self.stages
        # Stage 1
        for s in range(stages):
            pred = 0
            for mode in modes:
                sub_module = self.module_dict["s{}_{}".format(str(s + 1), mode)]

                pad = mode_pad_dict[mode]
                for r in [0, 1, 2, 3]:
                    if s + 1 == stages:
                        pred += torch.tanh(torch.rot90(
                            sub_module(F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')),
                            (4 - r) % 4, [2, 3]))*127
                    else:
                        pred += torch.tanh(torch.rot90(
                            sub_module(F.pad(torch.rot90(x, r, [2, 3]), (0, pad, 0, pad), mode='replicate')),
                            (4 - r) % 4, [2, 3]))
            # if s + 1 == stages:
            #     pred = torch.tanh(pred) * 127 * 4*3
            # else:
            #     pred = torch.tanh(pred + x) * 127

            if s + 1 == stages:
                avg_factor, bias, norm = len(modes), 0, 1
                x = round_func((pred / avg_factor) + bias)
                if phase == "train":
                    x = x / 255.0
            else:
                avg_factor, bias, norm = len(modes) * 4, 127, 255.0
                # x = round_func(torch.clamp((pred/avg_factor) + bias, 0, 255)) / norm
                x = round_func(torch.clamp((pred/avg_factor+x),0,1)*255.0)/norm
        return x
