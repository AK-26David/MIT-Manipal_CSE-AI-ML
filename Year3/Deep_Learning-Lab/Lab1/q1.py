import torch 
t1=torch.arange(9)
print(t1)
# Reshape a tensor
reshaped_t1=t1.reshape(3,3)
print(reshaped_t1)

# View a tensor
view_t1=t1.view(3,3)
print(view_t1)

# Stacking 2 tensors 

t2 = torch.rand(2,3)
t3 = torch.rand(2,3)
stacked_tensor = torch.stack([t2,t3],dim=0)
print(stacked_tensor)

# Squeezing a tensor 

t4=torch.rand(1,2,3)
print(t4)
squeezed_t4=torch.squeeze(t4,dim=0) # removes the dimension with size 1 at index 0 
print(squeezed_t4)


# Unsqueezing a tensor

t5=torch.rand(1,2,3)
print(t5)
unsqueezed_t5=torch.unsqueeze(t5,dim=0)  # adds a new dimension at index 0
print(unsqueezed_t5)
