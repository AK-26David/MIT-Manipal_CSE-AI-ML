import torch 
t1=torch.rand(2,2)
permuted_torch=t1.permute(1,0)
print(permuted_torch)