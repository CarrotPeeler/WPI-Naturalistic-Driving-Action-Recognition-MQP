import torch
import torch.nn as nn
import numpy as np
from fvcore.common.config import CfgNode

class MultiCamNoiseCropV2Prompter(nn.Module):
    def __init__(self, args):
        super(MultiCamNoiseCropV2Prompter, self).__init__()

        self.image_size = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size
        self.max_pad_size = args.PROMPT.PROMPT_SIZE if isinstance(args, CfgNode) else args.prompt_size
        self.crop_size = self.image_size - self.max_pad_size

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
            offset_left = int(np.random.randint(0, self.crop_size))
            offset_right = self.max_pad_size*2 - offset_left

            offset_up = int(np.random.randint(0, self.crop_size))
            offset_down = self.max_pad_size*2 - offset_up

            # calc number of each pad type (size 1 or 10) required to construct each side's pad
            num_size_10_pads_left = int(offset_left/10)*10
            num_size_1_pads_left = offset_left - num_size_10_pads_left

            num_size_10_pads_right = int(offset_right/10)*10
            num_size_1_pads_right = offset_right - num_size_10_pads_right
            
            num_size_10_pads_up = int(offset_up/10)*10
            num_size_1_pads_up = offset_up - num_size_10_pads_up

            num_size_10_pads_down = int(offset_down/10)*10
            num_size_1_pads_down = offset_down - num_size_10_pads_down

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
