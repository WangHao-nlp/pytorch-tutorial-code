import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 超参数
input_size = 1
output_size = 1
num_epochs = 2000
lr = 0.0001

np.random.seed(5)
np.random.rand(10, 1)
x_train = np.array([[1.5], [2.5], [3.5]], dtype=np.float32)
y_train = np.array([[6.5], [9.5], [12.5]], dtype=np.float32)
x_train = np.random.rand(10, 1).astype(np.float32)
y_train = np.random.rand(10, 1).astype(np.float32)

model = nn.Linear(input_size, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
    output = model(inputs)
    loss = criterion(output, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1)%5 == 0:
        print("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

predicted = model(torch.from_numpy(x_train)).detach().numpy()  # detach()阻断梯度计算
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend()
plt.show()




