import torch
import torch.nn as nn

class vgg_layer(nn.Module):
    #定义初始化函数
    def __init__(self, in_channels, out_channels):
        super(vgg_layer, self).__init__()#继承父类初始化函数
        self.conv1 = nn.Conv2d(in_channels, out_channels)#设置第一层卷积 channel即通道数
        self.conv2 = nn.Conv2d(in_channels, out_channels)#设置第二层卷积
        self.pool = nn.MaxPool2d(2)#池化层

    def forward(self, x):#前向传播
        x = self.conv1(x)#调用第一层卷积
        x = self.conv2(x)#调用第二层卷积
        x = self.pool(x)#调用池化层

        return x
    
class MyModel(nn.Module):
    def __init__(self, num_cls):#初始化模型函数
        super(MyModel, self).__init__()#继承父类初始化模型
        self.layer1 = vgg_layer(3, 64, 64)#卷积
        self.layer2 = vgg_layer(64, 128, 128)
        self.layer3 = vgg_layer(128, 256, 256)
        self.layer4 = vgg_layer(256, 512, 512)
        self.layer5 = vgg_layer(512, 512, 512)

class MyVgg(nn.Module):
    def __init__(self):
        super(MyVgg, self).__init__()
        self.layer1 = vgg_layer(3, 64)
        self.layer2 = vgg_layer(64, 128)
        self.layer3 = vgg_layer(128, 256)
        self.layer4 = vgg_layer(256, 512)
        self.layer5 = vgg_layer(512, 512)
        self.adapool = nn.AdaptiveAvgPool2d(7)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x