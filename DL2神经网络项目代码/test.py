import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import csv
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import time
from model import MyModel
import os
os.environ["KMP_DUPLICATE_lIB_OK"]="TRUE"

class CovidDatatest(Dataset):#用于加载数据集
    def __init__(self, file_path, mode="train"):#初始化,设置为训练模式  用mode来区分训练集还是测试集
        with open(file_path, "r") as f:#打开记录数据的文件
            ori_data = list(csv.reader(f))#将文件中的数据读取出来
            csv_data = np.array(ori_data[1:])[:, 1:].astype(float)    #不要第一列,去掉第一行第一列  将数据转化为numpy数组  astype是将数据转化为float类型

        if mode == "train":#训练集  
            indices = [i for i in range(len(csv_data))if i % 5 != 0]#逢5取1
            data = torch.tensor(csv_data[indices, :-1])  # 进入神经网络必须要张量tensor 把数据转化为张量
            self.y = torch.tensor(csv_data[indices, -1]) #y表示最后一列 用于训练
        elif mode == "val":#验证集
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]#逢5取0
            data = torch.tensor(csv_data[indices, :-1])  # 进入神经网络必须要张量tensor把数据转化为张量
            
            self.y = torch.tensor(csv_data[indices, -1])#y表示最后一列 用于训练
        else:
            indices = [i for i in range(len(csv_data))]#测试集
            data = torch.tensor(csv_data[indices])  # 进入神经网络必须要张量tensor把数据转化为张量
        self.data = (data-data.mean(dim=0, keepdim=True))/data.std(dim=0, keepdim=True)#标准化数据
        self.mode = mode   #表示当前是训练集还是测试集

    def __getitem__(self, idx):#使对象可以使用索引来访问数据集中的样本
        if self.mode != "test":#训练集和验证集的情况  需要数据和标签来计算损失以及更新参数
            return self.data[idx].float(), self.y[idx].float()#若为测试集，则转化模型数据x y类型为32位，变小一点，减少开销
        else:#测试集的情况
            return self.data[idx].float()

    def __len__(self):#使得数据加载器可以知道数据集的长度，从而能够正确地迭代数据集。
        return len(self.data)



#训练模型
def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, save_path):#传入参数 模型参数 训练数据 ______
    model = model.to(device)
    # epoch = 10
    plt_train_loss = []#就是用来记录所有训练轮次的loss
    plt_val_loss = []
    main_val_loss = 999999999   #记录最小的loss值，如果有小于该值的loss，就将loss替换为更小的

    #开始训练的地方
    for epoch in range(epochs):   #最主要的地方  训练每一轮loss
        train_loss = 0.0
        val_loss = 0.0
        start_time = time.time()    #用来计算训练时间

        model.train()       #模型调整为训练模式
        for batch_x, batch_y in train_loader:      #从训练集中去除一批数据x和y
            x, target = batch_x.to(device), batch_y.to(device) #放在gpu上训练
            pred = model(x)         #将x通过模型得出预测值
            train_bat_loss = loss(pred, target) #mse就是求两个y的平方差
            train_bat_loss.backward()#梯度回传
            optimizer.step()#起到更新训练模型的作用
            optimizer.zero_grad()#清除梯度堆积 为下一轮训练做准备
            train_loss += train_bat_loss.cpu().item()#张量没法在gpu上面跑，需要在cpu上面跑
        plt_train_loss.append(train_loss/train_loader.dataset.__len__())#train_loss是本次的轮次，要把它加载在所有loss里, 相加后去平均值  

        model.eval()#模型调整为测试模式，进行评估
        with torch.no_grad():#在模型中计算都会计算梯度，在验证集中只是看模型的效果，不可以积攒梯度
            for batch_x, batch_y in val_loader:#从验证数据集中获取一批数据 
                x, target = batch_x.to(device), batch_y.to(device)#将张量x,y移动到cpu上计算
                pred = model(x)#从训练模型中的出预测值
                val_bat_loss = loss(pred, target)#对于梯度回传，loss越小说明模型越好
                val_loss += val_bat_loss.cpu().item()#累加损失值
        plt_val_loss.append(val_loss/ val_loader.__len__())#记录每一轮的valloss
        if val_loss < main_val_loss:#val_loss为当前的损失值，每次选择比当前轮次以前最小的损失值还要小的损失值
            torch.save(model, save_path)#保存好的模型到save_path路径
            main_val_loss = val_loss#更新最小损失值
        # 打印出当前轮次训练的结果
        print("[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f"% \
              (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1])
              )#打印输出训练时间，训练损失值，验证损失值
    #plot用于画图
    plt.plot(plt_train_loss)#画出训练损失变化曲线图像
    plt.plot(plt_val_loss)#画出验证损失变化曲线图像
    plt.title("loss图")
    plt.legend(["train", "val"])
    plt.show()


#超参
device = "cuda" if torch.cuda.is_available() else "cpu"#若有gpu则选择cuda来跑模型，若没有gpu则选择cpu来跑模型
print(device)
#放置超参数
config = {
    "lr": 0.001,#学习率
    "epochs": 20,#训练轮次
    "momentum": 0.8,#惯性
    "save_path": "model_save/best_model.pth",#训练结果保存的路径
    "rel_path": "pred.csv"#预测值保存的路径
}#便于查看参数以及灵活修改


######################以上为函数准备工作，下面为训练集，验证集，测试集的定义转化为对象，用于函数调用
train_file = "covid.train.csv"#标注训练集的文件
test_file = "covid.test.csv"#标注测试集的文件
train_dataset = CovidDatatest(train_file, "train")#将部分数据集数据转为一个包含训练数据的训练集对象
val_dataset = CovidDatatest(train_file, "train")#将部分训练集数据转为一个包含验证数据的验证集对象
test_dataset = CovidDatatest(test_file, "train")#将测试集数据转为一个包含测试数据的测试集对象

batch_size = 16#设置每轮训练选取的数据量
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#把数据集传进来，shuffle就是起到打乱数据的作用
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#把数据集传进来，shuffle就是起到打乱数据的作用
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)#把数据集传进来，shuffle就是起到打乱数据的作用
model = MyModel(inDim=93).to(device)#定义模型，输入参数为维度，将模型移动到设备（cpu/gpu）上
loss = nn.MSELoss()#计算均方误差，用于回归任务
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])#随机梯度下降（优化器直接用官方的）
train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])#训练和验证模型，参数均为以上定义的对象，最后设置结果数据保存路径

#训练，验证，测试。  以下即最后一步：测试
#测试集
#评估函数，得出测试结果
def evaluate(save_path, test_loader, device, rel_path):
    #加载模型
    model = torch.load(save_path).to(device)#将训练出的数据转为张量传到设备上通过模型训练
    rel = []#定义一个结果的列表
    #让x通过模型预测出y
    with torch.no_grad():#不计算梯度
        for x in test_loader:#测试轮数
            pred = model(x.to(device))#将测试数据传到设备上通过模型得出预测值
            rel.append(pred)#将预测结果添加到列表中
    print(rel)

    with open(rel_path, "w", newline='') as f:#将预测结果保存到指定csv文件中，每行包含一个索引和对应的预测值
        csvWriter = csv.writerow(["id", "tested_positive"])
        for i, value in enumerate(rel):
            csvWriter.writerow([str(i), str(value)])
    print("文件已保存到{}".format((rel_path)))



# for batch_x, batch_y in train_dataset:
#     print(batch_x, batch_y)
# model = MyModel(inDim=93)
# predy = model(batch_x)
# file = pd.read_csv(train_file)
# print(file.head())



