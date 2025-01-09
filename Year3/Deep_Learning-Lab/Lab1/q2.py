import torch 
t1=torch.rand(3,3)
permuted_torch=t1.permute(1,0)
print(permuted_torch)