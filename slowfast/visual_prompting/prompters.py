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
from fvcore.common.config import CfgNode


class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.PROMPT.PROMPT_SIZE if isinstance(args, CfgNode) else args.prompt_size
        image_size = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size

        self.base_size = image_size - pad_size*2
        self.pad_up = nn.Parameter(torch.randn([1, 3, 1, pad_size, image_size]))
        self.pad_down = nn.Parameter(torch.randn([1, 3, 1, pad_size, image_size]))
        self.pad_left = nn.Parameter(torch.randn([1, 3, 1, image_size - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.randn([1, 3, 1, image_size - pad_size*2, pad_size]))

    def forward(self, x):
        # start with dims [B x T x C x H x W]
        base = torch.zeros(1, 3, 1, self.base_size, self.base_size).cuda()
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
        prompt = torch.cat(x.size(2) * [prompt], dim=2)
        prompt = torch.cat(x.size(0) * [prompt], dim=0)
        
        return [x + prompt] # pyslowfast models expect list of tensors as input
        


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size
        self.psize = args.PROMPT.PROMPT_SIZE if isinstance(args, CfgNode) else args.prompt_size
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
        self.crop_size = torch.tensor(args.image_size, dtype=torch.int16)
        self.cam_view_dict = {
            'Dashboard': 0,
            'Right_side_window': 1,
            'Rear_view': 2
        }

        # corresponding cam_view indices for parameter tensors: Dashboard = 0, Right_side_window = 1, Rear_view = 2
        self.resize = torch.nn.Parameter(torch.randint(low=self.crop_size, high=1024, size=(3,), dtype=torch.float32))
        self.y_offset = torch.nn.Parameter(torch.randint(low=0, high=32, size=(3,), dtype=torch.float32))
        self.x_offset = torch.nn.Parameter(torch.randint(low=0, high=32, size=(3,), dtype=torch.float32))

    def forward(self, x, cam_views):
        assert x.shape[3] == x.shape[4], f"input x does not have matching height and width dimensions; got ({x.shape[3]}, {x.shape[4]})"
        assert x.shape[3] >= self.crop_size.item(), "input x height and width dimensions are smaller than the desired crop size"
        assert x.shape[0] == len(cam_views), f"len of cam_views does not match batch size of x; expected {x.shape[0]}, got {len(cam_views)} instead"

        clamped_resize = torch.clamp(self.resize, min=x.shape[3], max=1024).type(torch.int16).cuda()
        clamped_y_offset = torch.clamp(self.y_offset, min=torch.zeros(1).cuda(), max=clamped_resize - self.crop_size).type(torch.int16).cuda()
        clamped_x_offset = torch.clamp(self.x_offset, min=torch.zeros(1).cuda(), max=clamped_resize - self.crop_size).type(torch.int16).cuda()

        prompted_clips = []

        for clip_idx in range(x.shape[0]):
            cam_view_idx = self.cam_view_dict[cam_views[clip_idx]]
            
            resized_clip = torch.nn.functional.interpolate(
                x[clip_idx],
                size=(clamped_resize[cam_view_idx], clamped_resize[cam_view_idx]),
                mode="bilinear",
                align_corners=False,
            )

            cropped_clip = resized_clip[
                :, 
                :, 
                clamped_y_offset[cam_view_idx] : clamped_y_offset[cam_view_idx] + self.crop_size, 
                clamped_x_offset[cam_view_idx] : clamped_x_offset[cam_view_idx] + self.crop_size
            ].unsqueeze(dim=0)

            prompted_clips.append(cropped_clip)

        prompt = torch.cat(prompted_clips, dim=0)

        return [prompt]
    

class MultiCamNoiseCropPrompter(nn.Module):
    def __init__(self, args):
        super(MultiCamNoiseCropPrompter, self).__init__()

        # Parameters below may need to be manually adjusted
        self.image_size = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size

        pad_up_size = {
            'Dashboard': 60,
            'Right_side_window': 20,
            'Rear_view': 60
        }

        pad_left_size = {
            'Dashboard': 60,
            'Right_side_window': 40,
            'Rear_view': 60
        }

        pad_down_size = {
            'Dashboard': 30,
            'Right_side_window': 0,
            'Rear_view': 30
        }

        pad_right_size = {
            'Dashboard': 30,
            'Right_side_window': 30,
            'Rear_view': 30
        }

        # Parameters below DO NOT need to be manually adjusted

        self.pad_up = nn.ParameterDict({
            'Dashboard':  torch.randn([3, 1, pad_up_size['Dashboard'], self.image_size]),
            'Right_side_window': torch.randn([3, 1, pad_up_size['Right_side_window'], self.image_size]),
            'Rear_view': torch.randn([3, 1, pad_up_size['Rear_view'], self.image_size]),
        })

        self.pad_down = nn.ParameterDict({
            'Dashboard':  torch.randn([3, 1, pad_down_size['Dashboard'], self.image_size]),
            'Right_side_window': torch.randn([3, 1, pad_down_size['Right_side_window'], self.image_size]),
            'Rear_view': torch.randn([3, 1, pad_down_size['Rear_view'], self.image_size]),
        })

        pad_heights = {
            'Dashboard': self.image_size - pad_up_size['Dashboard'] - pad_down_size['Dashboard'],
            'Right_side_window': self.image_size - pad_up_size['Right_side_window'] - pad_down_size['Right_side_window'],
            'Rear_view': self.image_size - pad_up_size['Rear_view'] - pad_down_size['Rear_view']
        }

        self.pad_left = nn.ParameterDict({
            'Dashboard': torch.randn([3, 1, pad_heights['Dashboard'], pad_left_size['Dashboard']]),
            'Right_side_window': torch.randn([3, 1, pad_heights['Right_side_window'], pad_left_size['Right_side_window']]),
            'Rear_view': torch.randn([3, 1, pad_heights['Rear_view'], pad_left_size['Rear_view']]),
        })

        self.pad_right = nn.ParameterDict({
            'Dashboard': torch.randn([3, 1, pad_heights['Dashboard'], pad_right_size['Dashboard']]),
            'Right_side_window': torch.randn([3, 1, pad_heights['Right_side_window'], pad_right_size['Right_side_window']]),
            'Rear_view': torch.randn([3, 1, pad_heights['Rear_view'], pad_right_size['Rear_view']]),
        })


    def forward(self, x, cam_views):
        assert x.shape[0] == len(cam_views), \
            f"len of cam_views does not match batch size of x; expected {x.shape[0]}, got {len(cam_views)} instead"
        
        clip_prompts = []

        for clip_idx in range(x.shape[0]):
            cam_view = cam_views[clip_idx]

            base_height = self.image_size - self.pad_up[cam_view].shape[2] - self.pad_down[cam_view].shape[2]
            base_width = self.image_size - self.pad_left[cam_view].shape[3] - self.pad_right[cam_view].shape[3]

            base = torch.zeros(3, 1, base_height, base_width).cuda()

            clip_prompt = torch.cat([self.pad_left[cam_view], base, self.pad_right[cam_view]], dim=3)
            clip_prompt = torch.cat([self.pad_up[cam_view], clip_prompt, self.pad_down[cam_view]], dim=2)
            clip_prompt = torch.cat(x.size(2) * [clip_prompt], dim=1).unsqueeze(dim=0)

            clip_prompts.append(clip_prompt)

        prompt = torch.cat(clip_prompts, dim=0)

        return [x + prompt]
    

class MultiCamPadPrompter(nn.Module):
    def __init__(self, args):
        super(MultiCamPadPrompter, self).__init__()
        pad_size = args.PROMPT.PROMPT_SIZE if isinstance(args, CfgNode) else args.prompt_size
        image_size = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size

        self.base_size = image_size - pad_size*2

        self.pad_up = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, pad_size, image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, pad_size, image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, pad_size, image_size]))  
        })
        self.pad_down = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, pad_size, image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, pad_size, image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, pad_size, image_size]))  
        })
        self.pad_left = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, image_size - pad_size*2, pad_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, image_size - pad_size*2, pad_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, image_size - pad_size*2, pad_size]))  
        })
        self.pad_right = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, image_size - pad_size*2, pad_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, image_size - pad_size*2, pad_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, image_size - pad_size*2, pad_size]))  
        })

    def forward(self, x, cam_views):
        assert x.shape[0] == len(cam_views), \
            f"len of cam_views does not match batch size of x; expected {x.shape[0]}, got {len(cam_views)} instead"
        
        clip_prompts = []

        base = torch.zeros(3, 1, self.base_size, self.base_size).cuda()

        for clip_idx in range(x.shape[0]):
            cam_view = cam_views[clip_idx]

            clip_prompt = torch.cat([self.pad_left[cam_view], base, self.pad_right[cam_view]], dim=3)
            clip_prompt = torch.cat([self.pad_up[cam_view], clip_prompt, self.pad_down[cam_view]], dim=2)
            clip_prompt = torch.cat(x.size(2) * [clip_prompt], dim=1).unsqueeze(dim=0)

            clip_prompts.append(clip_prompt)

        prompt = torch.cat(clip_prompts, dim=0)
        
        return [x + prompt] # pyslowfast models expect list of tensors as input
    

class MultiCamNoiseCropV2Prompter(nn.Module):
    def __init__(self, args):
        super(MultiCamNoiseCropV2Prompter, self).__init__()

        self.image_size = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size
        self.max_pad_size = args.PROMPT.PROMPT_SIZE if isinstance(args, CfgNode) else args.prompt_size
        self.crop_size = self.image_size - self.max_pad_size*2

        self.pad_up_size_1 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, 1, self.image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, 1, self.image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, 1, self.image_size]))  
        })
        self.pad_up_size_10 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, 10, self.image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, 10, self.image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, 10, self.image_size]))  
        })

        self.pad_down_size_1 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, 1, self.image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, 1, self.image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, 1, self.image_size]))  
        })
        self.pad_down_size_10 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, 10, self.image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, 10, self.image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, 10, self.image_size]))  
        })
        
        self.pad_left_size_1 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.crop_size, 1])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.crop_size, 1])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.crop_size, 1]))  
        })
        self.pad_left_size_10 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.crop_size, 10])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.crop_size, 10])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.crop_size, 10]))  
        })

        self.pad_right_size_1 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.crop_size, 1])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.crop_size, 1])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.crop_size, 1]))  
        })
        self.pad_right_size_10 = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.crop_size, 10])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.crop_size, 10])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.crop_size, 10]))  
        })


    def forward(self, x, cam_views):
        assert x.shape[0] == len(cam_views), \
            f"len of cam_views does not match batch size of x; expected {x.shape[0]}, got {len(cam_views)} instead"
        
        clip_prompts = []

        for clip_idx in range(x.shape[0]):
            cam_view = cam_views[clip_idx]

            # calc the pad size for left, right, up, down pads
            offset_right = int(np.random.randint(1, self.max_pad_size+1))
            offset_left = self.max_pad_size*2 - offset_right

            offset_down = int(np.random.randint(1, self.max_pad_size+1))
            offset_up = self.max_pad_size*2 - offset_down

            # calc number of each pad type (size 1 or 10) required to construct each side's pad
            num_size_10_pads_left = int(offset_left/10)
            num_size_1_pads_left = offset_left - num_size_10_pads_left*10

            num_size_10_pads_right = int(offset_right/10)
            num_size_1_pads_right = offset_right - num_size_10_pads_right*10
            
            num_size_10_pads_up = int(offset_up/10)
            num_size_1_pads_up = offset_up - num_size_10_pads_up*10

            num_size_10_pads_down = int(offset_down/10)
            num_size_1_pads_down = offset_down - num_size_10_pads_down*10

            # concat pad types (size 1 and 10) together to construct final pad for each side (left, right, up, down)
            pad_left = torch.cat(num_size_10_pads_left*[self.pad_left_size_10[cam_view]] + num_size_1_pads_left*[self.pad_left_size_1[cam_view]], dim=3)
            pad_right = torch.cat(num_size_1_pads_right*[self.pad_right_size_1[cam_view]] + num_size_10_pads_right*[self.pad_right_size_10[cam_view]], dim=3)

            pad_up = torch.cat(num_size_10_pads_up*[self.pad_up_size_10[cam_view]] + num_size_1_pads_up*[self.pad_up_size_1[cam_view]], dim=2)
            pad_down = torch.cat(num_size_1_pads_down*[self.pad_down_size_1[cam_view]] + num_size_10_pads_down*[self.pad_down_size_10[cam_view]], dim=2)

            # create base tensor of crop size which will connect the four sides' pads together
            base = torch.zeros(3, 1, self.crop_size, self.crop_size).cuda()

            # concat pads onto base tensor
            clip_prompt = torch.cat([pad_left, base, pad_right], dim=3)
            clip_prompt = torch.cat([pad_up, clip_prompt, pad_down], dim=2)
            clip_prompt = torch.cat(x.size(2) * [clip_prompt], dim=1).unsqueeze(dim=0)

            # append finalized prompt for single clip to list
            clip_prompts.append(clip_prompt)

        # concat prompts for all clips together into one tensor
        prompt = torch.cat(clip_prompts, dim=0)
        
        return [x + prompt] # pyslowfast models expect list of tensors as input


def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)


def crop(args):
    return CropPrompter(args)


def multi_cam_noisecrop(args):
    return MultiCamNoiseCropPrompter(args)


def multi_cam_noisecropv2(args):
    return MultiCamNoiseCropV2Prompter(args)


def multi_cam_padding(args):
    return MultiCamPadPrompter(args)
