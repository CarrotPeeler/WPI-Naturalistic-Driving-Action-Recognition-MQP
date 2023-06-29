import os
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, utils

# target_dim = 10
# dim = int(torch.randint(low=1, high=5, size=(1,)).item())
# data = torch.ones((1,3,1,dim,dim))
# # pad(left, right, top, bottom)
# new_data = torch.nn.functional.pad(input=data, pad=(0, target_dim-dim, 0, target_dim-dim), mode='constant', value=0)
# print(new_data.shape[3])

class Self_Attn(nn.Module):
    """ Self attention Layer -- Adapted from SAGAN Paper https://arxiv.org/pdf/1805.08318.pdf """
    def __init__(self, in_dim, activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = 32 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = 32 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X C X (N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x

        return out, attention

class Attn(nn.Module):
    def __init__(self):
        super(Attn, self).__init__()
        
        out_dim = 64

        self.l1 = nn.Sequential(
            nn.Conv2d(3, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )

        self.l2 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim*2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )

        out_dim *= 2

        self.l3 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim*2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )

        out_dim *= 2

        self.self_attn_1 = Self_Attn(out_dim, 'relu')

        self.l4 = nn.Sequential(
            nn.Conv2d(out_dim, out_dim*2, 4, 2, 1),
            nn.LeakyReLU(0.1)
        )

        out_dim *= 2

        self.self_attn_2 = Self_Attn(out_dim, 'relu')

        self.last = nn.Sequential(
            nn.Conv2d(out_dim, 1, 4)
        )

    def forward(self, x):
        assert x.shape[3] == x.shape[4], f"input x does not have matching height and width dimensions; got ({x.shape[3]}, {x.shape[4]})"
        # assert x.shape[0] == len(cam_views), f"len of cam_views does not match batch size of x; expected {x.shape[0]}, got {len(cam_views)} instead"

        x = x.permute(0, 2, 1, 4, 3) # (B X C X T X H X W) => (B X T X C X W X H) to match Self_Attn input dims

        prompt_clips = []

        for clip in x: # => (T X C X W X H) => wherever dim B appears below its actually dim T
            out = self.l1(clip)
            out = self.l2(out)
            out = self.l3(out)
            out, attn_1 = self.self_attn_1(out)
            out = self.l4(out)
            out, attn_2 = self.self_attn_2(out)
            out = self.last(out).unsqueeze(dim=0)

            prompt_clips.append(out)

        prompt = torch.cat(prompt_clips, dim=0)
        prompt = prompt.permute(0, 2, 1, 4, 3)

        return prompt
    
image = Image.open(os.getcwd() + "/slowfast/visual_prompting/images/originals/0_0_0.png")
transform = transforms.ToTensor()

frame = transform(image).unsqueeze(dim=1)
clip = torch.cat([frame]*16, dim=1).unsqueeze(dim=0)
clip_batch = torch.cat([clip]*16, dim=0).to("cuda")

attn_model = Attn().to("cuda")
out_clip_batch = attn_model(clip_batch)

utils.save_image(out_clip_batch[0].permute(1, 0, 2, 3)[0], os.getcwd() + "/slowfast/visual_prompting/images/self_attn/test.png")
