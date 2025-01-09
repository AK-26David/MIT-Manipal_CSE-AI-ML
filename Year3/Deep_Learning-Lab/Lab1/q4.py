import torch 
import numpy as np 
a = np.array([1,2,3,4])
t1=torch.from_numpy(a)
print(t1)
b=torch.Tensor.numpy(t1)
print(b)