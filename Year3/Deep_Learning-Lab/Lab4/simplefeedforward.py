import torch 
from torch import nn 
torch.manual_seed(42)
linear=nn.Linear(3,2)
print("Network Structure:\n",linear)
print("Weight:\n",linear.weight)
print("Bias:\n",linear.bias)
x=torch.tensor([1.0,2.0,3.0])
output=linear(x)
print("Input x :\n",x)
print("Output:\n",output)
y_manual=torch.matmul(x,linear.weight.t())+linear.bias
print(y_manual)