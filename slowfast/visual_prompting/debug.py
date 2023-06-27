import torch

# target_dim = 10
# dim = int(torch.randint(low=1, high=5, size=(1,)).item())
# data = torch.ones((1,3,1,dim,dim))
# # pad(left, right, top, bottom)
# new_data = torch.nn.functional.pad(input=data, pad=(0, target_dim-dim, 0, target_dim-dim), mode='constant', value=0)
# print(new_data.shape[3])

x = torch.rand((1,3,1,256,256)).cuda()
crop_size = torch.tensor(224, dtype=torch.int16).cuda()

resize = torch.nn.Parameter(torch.randint(low=crop_size, high=1024, size=(3,), dtype=torch.float32)).cuda()
y_offset = torch.nn.Parameter(torch.randint(low=0, high=32, size=(3,), dtype=torch.float32)).cuda()
x_offset = torch.nn.Parameter(torch.randint(low=0, high=32, size=(3,), dtype=torch.float32)).cuda()

# assert x.shape[3] == x.shape[4], f"input x does not have matching height and width dimensions; got ({x.shape[3]}, {x.shape[4]})"
# assert x.shape[3] >= crop_size.item(), "input x height and width dimensions are smaller than the desired crop size"

clamped_resize = torch.clamp(resize, min=x.shape[3], max=1024).type(torch.int16)
# clamped_y_offset = torch.clamp(y_offset, min=torch.zeros(1), max=clamped_resize - crop_size).type(torch.int16)
# clamped_x_offset = torch.clamp(x_offset, min=torch.zeros(1), max=clamped_resize - crop_size).type(torch.int16)

# new_clips = torch.nn.functional.interpolate(
#     input=x.squeeze(dim=0),
#     size=(clamped_resize, clamped_resize),
#     mode="bilinear",
#     align_corners=False,
# )

# crop = new_clips[
#     :, :, clamped_y_offset : clamped_y_offset + crop_size, clamped_x_offset : clamped_x_offset + crop_size
# ]

# print(crop.shape)
clamped_x_offset = torch.clamp(x_offset, min=torch.zeros(1).cuda(), max=clamped_resize - crop_size).type(torch.int16).cuda()
print(x_offset, clamped_x_offset)
print(x_offset[2], clamped_x_offset[2])