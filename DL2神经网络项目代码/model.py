import torch.nn as nn
class MyModel(nn.Module):#定义模型函数，用于生成模型来训练数据
    def __init__(self, inDim):#初始化模型
        super(MyModel, self).__init__()#初始化
        self.fc1 = nn.Linear(inDim, 64)#第一层全链接  需要激活函数激活
        self.relu1 = nn.ReLU()#使用激活函数relu
        self.fc2 = nn.Linear(64, 1)#第二层全链接 输出直接到一维  需要激活函数激活

    def forward(self, x):#模型前向过程 定义输入数据如何通过模型的各层进行计算得到输出结果
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        #会导致两维的x，所以要变成一维的 保证输出张量的形状符合预期
        if len(x.size()) > 1:
            return x.squeeze(1)#去掉一维的

        return x