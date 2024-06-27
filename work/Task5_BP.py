from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import torch.optim as optim
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

batch_size = 32
input_size = 6  # 6个特征
output_size = 5  # 5个类别
num_epochs = 100
lr = 1e-6
label_encoder = LabelEncoder()

def dataLoad():
    # 创建数据集和数据加载器
    data = pd.read_csv('../NBA_Season_Stats.csv')
    X_columns = ['Age', 'PTS', 'TRB', 'AST', 'STL', 'BLK']
    X = data[X_columns]
    y = data['Pos']
    y = label_encoder.fit_transform(y)
    X = X.to_numpy()
    X = torch.tensor(X).type(torch.FloatTensor)
    y = torch.tensor(y).type(torch.LongTensor)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # 将标签转换为one-hot编码
    y_train_one_hot = torch.eye(output_size)[y_train]
    # 创建数据加载器
    train_loader = DataLoader(list(zip(X_train, y_train_one_hot)), batch_size=32, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=32, shuffle=True)
    return train_loader, test_loader

train_loader, test_loader = dataLoad()

# 定义BP神经网络模型
class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(BPNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 512)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

# 初始化网络、损失函数和优化器
model = BPNeuralNetwork(input_size)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def predict(model, data_loader):
    model.eval()  # 将模型设置为评估模式
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 在评估过程中不计算梯度
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels

def calculate_accuracy(preds, labels):
    # 确保preds和labels是数组形式，并且长度相同
    if len(preds) != len(labels):
        raise ValueError("The length of preds and labels must be the same")
    # 逐元素比较preds和labels，生成布尔数组
    correct = (np.array(preds) == np.array(labels)).sum()
    accuracy = correct / len(labels)
    return accuracy

# 训练模型
loss_list = []
preds, labels = [], []
for epoch in range(num_epochs):
    for inputs, labels in train_loader:  # 假设train_loader是数据加载器
        optimizer.zero_grad()  # 清除之前的梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重
    accuracy = 0

    preds, labels = predict(model, test_loader)
    accuracy = calculate_accuracy(preds, labels)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}, accuracy:{accuracy:.3f}')
    loss_list.append(loss.item())

def drawLoss():
    plt.plot(range(0, num_epochs), loss_list, 'o-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('../result/Loss.png')
    plt.clf()

def drawConfusionMatrix():
    conf_matrix = confusion_matrix(preds, labels)
    # 绘制混淆矩阵
    ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(values_format='.0f', cmap='Blues')  # 可以根据需要调整参数
    plt.savefig('../result/confusion_matrix_BP.png')
    plt.clf()

drawLoss()
drawConfusionMatrix()
