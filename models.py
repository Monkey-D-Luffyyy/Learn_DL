import torch
import torch.nn as nn


# 自定义神经网络模型
class QYnet(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入是28x28的灰度图像，输出是256个神经元
        self.fc2 = nn.Linear( 256, 128)  # 第二层，全连接层，输入256个神经元，输出128个神经元
        self.fc3 = nn.Linear(128, 64)  # 第三层，全连接层，输入128个神经元，输出64个神经元
        self.fc4 = nn.Linear(64, 10)  # 最后一层，全连接层，输入64个神经元，输出10个类别

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)  # 展平数据，方便进行全连接
        x = torch.relu(self.fc1(x))  # 第一层 + ReLU激活
        x = torch.relu(self.fc2(x))  # 第二层 + ReLU激活
        x = torch.relu(self.fc3(x))  # 第三层 + ReLU激活
        x = self.fc4(
            x
        )  # 第四层（输出层，不需要激活函数）  nn.CrossEntropyLoss会在内部进行softmax操作
        # x = torch.softmax(x, dim=1)  # 使用 softmax 激活函数1代表对类别维度进行softmax
        return x
