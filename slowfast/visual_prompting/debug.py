import torch

# target_dim = 10
# dim = int(torch.randint(low=1, high=5, size=(1,)).item())
# data = torch.ones((1,3,1,dim,dim))
# # pad(left, right, top, bottom)
# new_data = torch.nn.functional.pad(input=data, pad=(0, target_dim-dim, 0, target_dim-dim), mode='constant', value=0)
# print(new_data.shape[3])

in_dim = 3
x = torch.randn((16,in_dim,224,224))
out = torch.nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8 , kernel_size= 1)
print(out(x).shape)