import torch
x = torch.tensor([3.0],requires_grad=True)
y = torch.tensor([2.0],requires_grad=True)
z=torch.log(3*y**2+5*x*y)
z.backward()
print(x.grad)
print(y.grad)