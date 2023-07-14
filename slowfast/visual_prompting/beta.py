import os
import torch
import torch.nn as nn
import numpy as np
from fvcore.common.config import CfgNode
from torchvision.utils import save_image

class MultiCamNoiseCropV3Prompter(nn.Module):
    def __init__(self, args):
        super(MultiCamNoiseCropV3Prompter, self).__init__()

        self.image_size = args.DATA.TRAIN_CROP_SIZE if isinstance(args, CfgNode) else args.image_size
        self.max_pad_size = args.PROMPT.PROMPT_SIZE if isinstance(args, CfgNode) else args.prompt_size
        self.crop_size = self.image_size - self.max_pad_size*2

        self.pad_up = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.max_pad_size*2, self.image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.max_pad_size*2, self.image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.max_pad_size*2, self.image_size]))  
        })
    
        self.pad_down = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.max_pad_size, self.image_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.max_pad_size, self.image_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.max_pad_size, self.image_size]))  
        })
        
        self.pad_left = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.crop_size, self.max_pad_size*2])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.crop_size, self.max_pad_size*2])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.crop_size, self.max_pad_size*2]))  
        })

        self.pad_right = nn.ParameterDict({
            'Dashboard': nn.Parameter(torch.randn([3, 1, self.crop_size, self.max_pad_size])),
            'Right_side_window': nn.Parameter(torch.randn([3, 1, self.crop_size, self.max_pad_size])),
            'Rear_view': nn.Parameter(torch.randn([3, 1, self.crop_size, self.max_pad_size]))  
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

            # perform interpolation 
            pad_left = torch.nn.functional.interpolate(
                self.pad_left[cam_view],
                size=(self.pad_left[cam_view].shape[2], offset_left),
                mode="bilinear",
                align_corners=False,
            )

            pad_right = torch.nn.functional.interpolate(
                self.pad_right[cam_view],
                size=(self.pad_right[cam_view].shape[2], offset_right),
                mode="bilinear",
                align_corners=False,
            )

            pad_up = torch.nn.functional.interpolate(
                self.pad_up[cam_view],
                size=(offset_up, self.pad_up[cam_view].shape[3]),
                mode="bilinear",
                align_corners=False,
            )

            pad_down = torch.nn.functional.interpolate(
                self.pad_down[cam_view],
                size=(offset_down, self.pad_down[cam_view].shape[3]),
                mode="bilinear",
                align_corners=False,
            )

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
    

class MultiCamPadV2Prompter(nn.Module):
    def __init__(self, args):
        super(MultiCamPadV2Prompter, self).__init__()
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
    
# max_size = 30
# randten = torch.randn([1, 3, 224, max_size*2])
# offset_right = int(np.random.randint(1, max_size+1)) 
# offset_left = max_size*2 - offset_right

# pad_left = torch.nn.functional.interpolate(
#         randten,
#         size=(randten.shape[2], offset_left),
#         mode="bilinear",
#         align_corners=False,
#     )

# save_image(randten, os.getcwd() + f"/visual_prompting/images/self_attn/orig.png")
# save_image(pad_left, os.getcwd() + f"/visual_prompting/images/self_attn/interp.png")