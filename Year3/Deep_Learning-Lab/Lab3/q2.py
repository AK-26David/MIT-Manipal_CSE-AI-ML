import torch
import numpy as np
from torch.autograd import grad
import matplotlib.pyplot as plt

inp_x = np.array([2, 4])
inp_y = np.array([20, 40])

x = torch.tensor(inp_x)
y = torch.tensor(inp_y)
b = torch.tensor(1., requires_grad=True)
w = torch.tensor(1., requires_grad=True)
learning_rate = torch.tensor(0.001)
loss_list = []

for epochs in range(2):
    loss = 0.0
    for j in range(len(x)):
        a = w * x[j]
        y_p = a + b
        loss += (y[j] - y_p) ** 2
    loss = loss / len(x)
    loss_list.append(loss.item())
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad

    print(w.grad, b.grad)
    w.grad.zero_()
    b.grad.zero_()
    print("w = {},b = {},loss = {}".format(w, b, loss))


def analytical(x, y, w, b):
    for epoch in range(2):
        loss = 0.0
        for j in range(len(x)):
            y_p = x[j] * w + b
            loss += (y_p - y[j]) ** 2
        loss = loss / len(x)
        wgrad, bgrad = 0, 0
        for j in range(len(x)):
            wgrad += (y_p - y[j]) * (x[j])
            bgrad += (y_p - y[j]) * 2
        w -= 0.001 * wgrad * 2 / len(x)
        b -= 0.001 * bgrad * 2 / len(x)
        print(f"w = {w}, b= {b}, loss = {loss}")
    return loss


print("Analytical Solution")
analytical(inp_x, inp_y, 1, 1)