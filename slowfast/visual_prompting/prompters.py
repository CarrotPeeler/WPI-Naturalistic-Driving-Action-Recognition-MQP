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
    

class CropPrompter(nn.Module):
    def __init__(self, args):
        super(CropPrompter, self).__init__()
        self.crop_size = args.image_size
        self.resize = nn.Parameter(torch.tensor(256.)) #(torch.randint(low=self.crop_size, high=1024, size=(1,), dtype=torch.float16, requires_grad=True))
        self.y_offset = nn.Parameter(torch.randint(low=0, high=32, size=(1,), dtype=torch.float16))
        self.x_offset = nn.Parameter(torch.randint(low=0, high=32, size=(1,), dtype=torch.float16))

    def forward(self, x):
        clamped_size = int(max(self.crop_size, min(1024, self.resize.data.item())))
        clamped_y_offset = int(max(0, min(abs(x.shape[3] - clamped_size), self.y_offset.data.item())))
        clamped_x_offset = int(max(0, min(abs(x.shape[4] - clamped_size), self.x_offset.data.item())))

        print(f"size: {clamped_size} y: {clamped_y_offset} x: {clamped_x_offset}")

        crops = []

        for clip_idx in range(x.shape[0]):
            
            resized_images = torch.nn.functional.interpolate(
                x[clip_idx],
                size=(clamped_size, clamped_size),
                mode="bilinear",
                align_corners=False,
            )

            crop = resized_images[
                :, :, clamped_y_offset : clamped_y_offset + self.crop_size, clamped_x_offset : clamped_x_offset + self.crop_size
            ]

            crop = crop.unsqueeze(dim=0) # 3 x 16 x 224 x 224 => 1 x 3 x 16 x 224 x 224
            crops.append(crop)

        prompt = torch.cat(crops, dim=0) # => 16 x 3 x 16 x 224 x 224)

        return [prompt]
    

class NoiseCropPrompter(nn.Module):
    def __init__(self, args):
        super(NoiseCropPrompter, self).__init__()
        self.crop_size = args.image_size
        self.target_resize = 512
        

    def forward(self, x):
        
        resized_clips = []

        for clip_idx in range(x.shape[0]):
            
            resized_clip = torch.nn.functional.interpolate(
                x[clip_idx],
                size=(self.target_resize, self.target_resize),
                mode="bilinear",
                align_corners=False,
            )

            resized_clip = resized_clip.unsqueeze(dim=0) # 3 x 16 x 224 x 224 => 1 x 3 x 16 x 224 x 224
            resized_clips.append(resized_clip)

        resized_clips_tensor = torch.cat(resized_clips, dim=0) # => 16 x 3 x 16 x 224 x 224)

        height = x.shape[3]
        width = x.shape[4]

        y_offset = 0
        if height > self.crop_size:
            y_offset = torch.randint(0, height - self.crop_size, size=(1,), dtype=torch.int16)

        x_offset = 0
        if width > self.crop_size:
            x_offset = torch.randint(0, width - self.crop_size, size=(1,), dtype=torch.int16)



def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)


def crop(args):
    return CropPrompter(args)


def noise_crop(args):
    return NoiseCropPrompter(args)