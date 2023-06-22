"""
MIT License

Copyright (c) 2022 Hyojin Bahng

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import numpy as np


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 1, 3, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 1, 3, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 1, 3, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        # start with dims [B x T x C x H x W]
        base = torch.zeros(1, 1, 3, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
        prompt = torch.cat(x.size(2) * [prompt], dim=1)
        prompt = torch.cat(x.size(0) * [prompt], dim=0)

        # permute [B x T x C x H x W] => [B x C x T x H x W] to match input tensor shape
        prompt = prompt.permute(0, 2, 1, 3, 4)
        
        return [x + prompt] # pyslowfast models expect list of tensors as input


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, 1, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, 1, self.isize, self.isize]).cuda()
        prompt[:, :, :, :self.psize, :self.psize] = self.patch

        return [x + prompt]


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.randn([1, 3, 1, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize - self.psize)
        y_ = np.random.choice(self.isize - self.psize)

        prompt = torch.zeros([1, 3, 1, self.isize, self.isize]).cuda()
        prompt[:, :, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch

        return [x + prompt]


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)