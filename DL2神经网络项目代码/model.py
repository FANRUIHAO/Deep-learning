import torch.nn as nn

#对模型进行改造（feature_dim, all_feature
class MyModel(nn.Module):
    def __init__(self, inDim):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(inDim, 100)#数据进入模型维度   第一层全链接  需要激活函数激活
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(100, 1)#第二层全链接 输出直接到一维  需要激活函数激活

    def forward(self, x):#模型前向过程
        x = self.fc1(x) #全链接
        x = self.relu1(x) #激活函数
        x = self.fc2(x) #全链接

        #会导致两维的x，难以转换为概率，所以需要将其转换为一维
        if len(x.size()) > 1:
            return x.squeeze(1)#去掉一维的

        return x