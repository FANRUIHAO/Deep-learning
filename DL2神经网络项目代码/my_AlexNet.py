import torch
import torch.nn as nn


class vgg_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(vgg_layer, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel)#第一个卷积
        self.conv2 = nn.Conv2d(out_channel, out_channel)#第一个卷积
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):#模型向前
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)

        return x


class MyVgg(nn.Module):
    def __init__(self):
        super(MyVgg, self).__init__()

        self.layer1 = vgg_layer(3, 64, 64)

        self.layer2 = vgg_layer(64, 128, 128)

        self.layer3 = vgg_layer(128, 256, 256)

        self.layer4 = vgg_layer(256, 512, 512)

        self.layer5 = vgg_layer(512, 512, 512)
        self.adapool = nn.AdaptiveAvgPool2d(7)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(25088, 4896)

    #VGG
class myModel(nn.Module):
     def __init__(self, num_cls):
         super(myModel, self).__init__()
         self.layer1 = vgg_layer(3,64,64)
         self.layer1 = vgg_layer(3, 64, 64)
         self.layer1 = vgg_layer(3, 64, 64)
         self.layer1 = vgg_layer(3, 64, 64)
         self.layer1 = vgg_layer(3, 64, 64)