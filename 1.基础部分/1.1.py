import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms


# 1.基础求导例子1
x = torch.tensor(1., requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)

y = w * x + b
y.backward()

print(x.grad)
print(w.grad)
print(b.grad)

# 2.基础求导例子2
x = torch.rand(10, 3)
y = torch.rand(10, 2)

liner = nn.Linear(3, 2)  # 全连接层
print("w: ", liner.weight)
print("b: ", liner.bias)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(liner.parameters(), lr=0.01)

pred = liner(x)

loss = criterion(pred, y)
print('loss: ', loss.item())

loss.backward()

print("dl/dw: ", liner.weight.grad)
print("dl/db: ", liner.bias.grad)

optimizer.step()

pred = liner(x)
loss = criterion(pred, y)

print('loss after 1 step optimizer: ', loss.item())

# 3.numpy to tensor to numpy
x = np.array([[1, 2], [3, 4]])
y = torch.from_numpy(x)
z = y.numpy()
print("x: ", x)
print("y: ", y)
print("z: ", z)

# 4.输入管道
train_dataset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                             transform=transforms.ToTensor(), download=True)

image, label = train_dataset[0]
print(image.size())
print(label)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64, shuffle=True)
data_iter = iter(train_loader)
image, label = data_iter.next()
count = 1
# for image, label in train_loader:
#     count += 1
# print(count)

# 5.一般数据输入管道


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x_data, y_data):
        self.data_tensor = x_data
        self.target_tensor = y_data

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


data_tensor = torch.randn(4, 3)
target_tensor = torch.rand(4)
tensor_dataset = TensorDataset(data_tensor, target_tensor)
print(tensor_dataset[1])
print(len(tensor_dataset))
loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=2, shuffle=True,  num_workers=0)
# 以for循环形式输出
for data, target in loader:
    print(data, target)

# 6.预训练模型
resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False

resnet.fc = nn.Linear(resnet.fc.in_features,100)
image = torch.rand(64, 3, 224, 244)
outputs = resnet(image)
print(outputs.size())

# 7.保存模型
# 保存整个模型
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')
# 保存模型参数（推荐）
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))