import torch
import torch.nn.functional as F

image = torch.rand(6 ,6)
print("image=", image)
# Add a new dimension along 0th dimension
# i.e. (6,6) becomes (1,6,6). This is because
# pytorch expects the input to conv2D as 4d tensor
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
image = image.unsqueeze(dim=0)
print("image.shape=", image.shape)
print("image=", image)
kernel = torch.ones(3 ,3)
# kernel = torch.rand(3,3)
print("kernel=", kernel)
kernel = kernel.unsqueeze(dim=0)
kernel = kernel.unsqueeze(dim=0)

def out_dim(in_shape ,stride ,padding ,kernel_shape):
    out_shape = [0 for i in range(4)]
    for dim in range(len(in_shape)):
        out_shape[dim] = (in_shape[dim] + 2* padding - kernel_shape[dim]) // stride + 1
    return out_shape


# Stride 1 Padding 0
outimage = F.conv2d(image, kernel, stride=1, padding=0)
print("outimage=", outimage)
print("Dimension of output image S-1 P-0: ", outimage.shape)
print("Manually dim of output S-1 P-0: ", out_dim(image.shape, 1, 0, kernel.shape))

# Stride 1 Padding 1
outimage = F.conv2d(image, kernel, stride=1, padding=1)
print("Dimension of output image S-1 P-1:", outimage.shape)
print("Manually dim of output S-1 P-1: ", out_dim(image.shape, 1, 1, kernel.shape))

# Stride 1 Padding 2
outimage = F.conv2d(image, kernel, stride=1, padding=2)
print("Dimension of output image S-1 P-2:", outimage.shape)
print("Manually dim of output S-1 P-2: ", out_dim(image.shape, 1, 2, kernel.shape))

# Stride 2 Padding 1
outimage = F.conv2d(image, kernel, stride=2, padding=1)
print("Dimension of output image S-2 P-1: ", outimage.shape)
print("Manually dim of output S-2 P-1: ", out_dim(image.shape, 2, 1, kernel.shape))

# Stride 3 Padding 1
outimage = F.conv2d(image, kernel, stride=3, padding=1)
print("Dimension of output image S-2 P-1:", outimage.shape)
print("Manually dim of output S-3 P-1: ", out_dim(image.shape, 3, 1, kernel.shape))

print("Number of Learnable Parameters = 9")

