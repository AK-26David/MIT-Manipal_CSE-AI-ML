import torch 
import math
x=torch.tensor([1.0],requires_grad=True)
u=-torch.sin(x)
u.retain_grad()
v=-2*x
v.retain_grad()
w=-x*x
w.retain_grad()
y=u+v+w
y.retain_grad()
f=torch.exp(y)
f.retain_grad()
f.backward()
print(f.grad)
print(y.grad.item())
print(w.grad.item())
print(v.grad.item())
print(u.grad.item())


