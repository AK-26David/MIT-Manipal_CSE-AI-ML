import torch 
t1=torch.randint(0,9,(3,3,3))
t2=torch.randint(0,9,(3,3,3))
t3=torch.bmm(t1,t2)
print(t3)