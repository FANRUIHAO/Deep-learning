import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, inDim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(inDim, 64)#第一层全链接  需要激活函数激活
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)#第二层全链接 输出直接到一维  需要激活函数激活

    def forward(self, x):#模型前向过程
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)

        #会导致两维的x，所以要变成一维的
        if len(x.size()) > 1:
            return x.squeeze(1)#去掉一维的

        return x