import torch 
t1=torch.randint(0,10,(7,7))
t2=torch.randint(0,10,(1,7))
t3=torch.matmul(t1,t2.T)
print(t3)