
import torch
import numpy as np
from torch.autograd import Variable

x = Variable(torch.tensor
    ([12.4 ,14.3 ,14.5 ,14.9 ,16.1 ,16.9 ,16.5 ,15.4 ,17.0 ,17.9 ,18.8 ,20.3 ,22.4 ,19.4 ,15.5 ,16.7 ,17.3 ,18.4 ,19.2
     ,17.4 ,19.5 ,19.7 ,21.2]))
y = Variable(torch.tensor
    ([11.2 ,12.5 ,12.7 ,13.1 ,14.1 ,14.8 ,14.4 ,13.4 ,14.9 ,15.6 ,16.4 ,17.7 ,19.6 ,16.9 ,14.0 ,14.6 ,15.1 ,16.1 ,16.8
     ,15.2 ,17.0 ,17.2 ,18.6]))

class RegressionModel(torch.nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.linear = torch.nn.Linear(23, 23)

    def forward(self ,x):
        return self.linear(x)

model = RegressionModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.0032)

for epochs in range(100):
    pred_y = model(x)
    loss = criterion(pred_y, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print("loss ={}".format(loss.item()))

