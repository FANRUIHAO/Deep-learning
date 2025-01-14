import random
import torch
import torch.nn as nn
import numpy as np
import os
from PIL import Image #读取图片数据
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm#用于运行程序时显示数据读取进度条，避免忙等
from torchvision import transforms
import time
import matplotlib.pyplot as plt
from model_utils.model import initialize_model

def seed_everything(seed):#设置随机种子 确保代码的可重复性和结果的可复现性  是的结果更加稳定，便于调试
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False #控制卷积算法 用false保证稳定性
    torch.backends.cudnn.deterministic = True#保证结果的确定性
    random.seed(seed)#打乱数据
    np.random.seed(seed)#用于大规模数据处理
    os.environ['PYTHONHASHSEED'] = str(seed)#设置哈希种子，它控制哈希函数的随机性，确保在每次运行时相同的输入值能得到相同的哈希值

#################################################################
seed_everything(0)
###############################################
HW = 224 #设置图片的高度和宽度

train_transform = transforms.Compose(#训练数据增强
    [
        transforms.ToPILImage(),   #224， 224， 3模型  ：3, 224, 224
        transforms.RandomResizedCrop(224),#对图像进行随机裁剪和缩放 最终输出一个固定大小的图像
        transforms.RandomRotation(50),#对图像进行随机旋转
        transforms.ToTensor()#将图像转化为张量
    ]
)

val_transform = transforms.Compose(#验证数据增强
    [
        transforms.ToPILImage(),   #224， 224， 3模型  ：3, 224, 224
        transforms.ToTensor()#将图像转化为张量
    ]
)

class food_Dataset(Dataset):#数据集初始化定义 对其作一定的处理
    def __init__(self, path, mode="train"):#初始化
        self.mode = mode#模式
        if mode == "semi":#半监督模式
            self.X = self.read_file(path)#调用方法读取文件中的图片装入对象中
        else:#全监督
            self.X, self.Y = self.read_file(path)#从文件读取数据装入对象中
            self.Y = torch.LongTensor(self.Y)  #标签转为长整形

        if mode == "train":#训练模式
            self.transform = train_transform#使用数据增强
        else:
            self.transform = val_transform#使用数据增强

    def read_file(self, path):#读取文件(即图片)
        if self.mode == "semi":#半监督的情况
            file_list = os.listdir(path)#存储指定目录下的所有文件和目录的名字
            xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)#规定具有特殊形状的数组  第一个参数为文件个数 |第二个参数为图片的高度和宽度| 第三个参数为通道数
            # 列出文件夹下所有文件名字
            for j, img_name in enumerate(file_list):#将存储的文件名字和索引值对应起来然后遍历
                img_path = os.path.join(path, img_name)#将文件名和路径连接起来
                img = Image.open(img_path)#打开图片
                img = img.resize((HW, HW))#调整图片大小
                xi[j, ...] = img#将图片装入数组中
            print("读到了%d个数据" % len(xi))#打印读取到了多少个文件
            return xi#返回图片数据
        else:#全监督的情况
            for i in tqdm(range(11)):#显示读取文件的进度条
                file_dir = path + "/%02d" % i#用于拼接路径
                file_list = os.listdir(file_dir)#存储指定目录下的所有文件和目录的名字

                xi = np.zeros((len(file_list), HW, HW, 3), dtype=np.uint8)#规定具有特殊形状的数组  |第一个参数为文件个数 |第二、三个参数为图片的高度和宽度| 第四个参数为通道数
                yi = np.zeros(len(file_list), dtype=np.uint8)#创建一维数组 并且初始化为0 用于存储标签

                # 列出文件夹下所有文件名字
                for j, img_name in enumerate(file_list):#将存储的文件名字和索引值对应起来然后遍历
                    img_path = os.path.join(file_dir, img_name) #将文件名和路径连接起来
                    img = Image.open(img_path) #打开图片
                    img = img.resize((HW, HW)) #调整图片高度和宽度
                    xi[j, ...] = img #将图片装入数组x中
                    yi[j] = i #将标签装入数组y中

                if i == 0: #第一次读取数据时
                    X = xi #将xi赋值给X 
                    Y = yi #将yi赋值给Y
                else: #第一次之后的数据读取
                    X = np.concatenate((X, xi), axis=0) #将xi和X合并 用于存储所有的图片数据
                    Y = np.concatenate((Y, yi), axis=0) #将yi和Y合并 用于存储所有的标签数据
            print("读到了%d个数据" % len(Y))
            return X, Y

    def __getitem__(self, item): #用于索引数据集 便于取出数据
        if self.mode == "semi":
            return self.transform(self.X[item]), self.X[item]#可以任意规定要返回的内容 此处返回了图片和图片的数据（半监督中可以返回一个x用于预测，返回一个x加入数据集）
        else:
            return self.transform(self.X[item]), self.Y[item]

    def __len__(self):#用于获取数据集的大小来确定迭代次数
        return len(self.X)

class semiDataset(Dataset):#半监督数据集  用到的是val_transform
    def __init__(self, no_label_loder, model, device, thres=0.99):
        x, y = self.get_label(no_label_loder, model, device, thres)#获取原始数据以及标签
        if x == []:#进行判断semidataset是否加载到了有效数据
            self.flag = False#用flag==false来表示semidataset没有加载到有效数据
        else:
            self.flag = True#用flag==true来表示semidataset加载到了有效数据
            self.X = np.array(x) #将数据转为数组
            self.Y = torch.LongTensor(y) #将标签转为长整形
            self.transform = train_transform #使用数据增强 用于训练
    def get_label(self, no_label_loder, model, device, thres):#获得标签
        model = model.to(device) #将模型放到设备上
        pred_prob = [] #存储预测概率
        labels = [] #存储标签
        x = []  #存储图片 用于返回
        y = [] #存储标签 用于返回
        soft = nn.Softmax() #使用softmax函数 用于将输出转为概率
        with torch.no_grad(): #取消梯度计算 用于测试集and验证集  
            for bat_x, _ in no_label_loder: #遍历无标签数据集 
                bat_x = bat_x.to(device) #将数据放到设备上
                pred = model(bat_x) #进行预测
                pred_soft = soft(pred) #将输出转为概率
                pred_max, pred_value = pred_soft.max(1) #取出概率最大的值和索引
                pred_prob.extend(pred_max.cpu().numpy().tolist()) #将最大的概率转为列表
                labels.extend(pred_value.cpu().numpy().tolist()) #将预测值转为列表

        for index, prob in enumerate(pred_prob): #遍历预测概率 用于筛选出概率大于阈值的数据
            if prob > thres: #如果概率大于阈值
                x.append(no_label_loder.dataset[index][1])   #调用到原始的getitem 将数据装入x中
                y.append(labels[index]) #将标签装入y中 这个标签就是预测的标签
        return x, y

    def __getitem__(self, item):    #对数据集进行索引 便于取出数据
        return self.transform(self.X[item]), self.Y[item] #进行数据增强 然后返回数据
    def __len__(self):  #用于获取数据集的大小来确定迭代次数
        return len(self.X)

def get_semi_loader(no_label_loder, model, device, thres): #获取半监督数据集
    semiset = semiDataset(no_label_loder, model, device, thres) #定义对象调用semiDataset计算出并存储半监督数据集
    if semiset.flag == False: #如果发现semidataset没有获得到有效数据
        return None #返回none
    else:#有加载到有效数据就返回semidataset
        semi_loader = DataLoader(semiset, batch_size=16, shuffle=False) #将semidataset转为dataloader
        return semi_loader



#对于模型:先定义 再初始化 最后训练和验证
#初始化时先定义卷积层要使用的 卷积、归一化、激活函数、池化 对象
#然后再创建四层卷积层对象 将定义 将定义的卷积 归一化 激活函数 池化作为参数传入四层卷积层对象中
#卷积层之间特征大小逐渐减小 通道数逐渐增加
#在定义向前传播函数所用的卷积、归一化、激活函数、池化对象
#以上准备工作都是用于forward函数的调用
#向前传播整体过程为:卷积 归一化 激活函数 池化 三层卷积层 池化 拉直 全连接层 激活函数 全连接层
class myModel(nn.Module):#定义模型
    def __init__(self, num_class):#初始化
        super(myModel, self).__init__() #继承nn.Module的初始化
        #3 *224 *224  -> 512*7*7 -> 拉直 ->全连接分类
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)    #卷积 64*224*224
        self.bn1 = nn.BatchNorm2d(64)  #归一化
        self.relu = nn.ReLU() #激活函数
        self.pool1 = nn.MaxPool2d(2)   # 规定池化窗口大小 然后取该区域的最大值作为输出，并将图像的尺寸缩小   64*112*112

        self.layer1 = nn.Sequential( #卷积层1 用于提取特征
            nn.Conv2d(64, 128, 3, 1, 1),    #卷积 128*112*112
            nn.BatchNorm2d(128), #归一化
            nn.ReLU(), #激活函数
            nn.MaxPool2d(2)   # 池化 128*56*56
        )
        self.layer2 = nn.Sequential( #卷积层2 用于提取特征
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)   #256*28*28
        )
        self.layer3 = nn.Sequential( #卷积层3 用于提取特征
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)   #512*14*14
        )

        self.pool2 = nn.MaxPool2d(2)    #池化 512*7*7
        self.fc1 = nn.Linear(25088, 1000)   # 全连接层 25088-1000
        self.relu2 = nn.ReLU() #激活函数
        self.fc2 = nn.Linear(1000, num_class)  #全连接层 1000-11

    def forward(self, x):#向前传播 此函数规定了数据在模型里向前（训练、预测）的过程
        x = self.conv1(x) #卷积
        x = self.bn1(x) #归一化
        x = self.relu(x) #激活函数
        x = self.pool1(x) #池化
        x = self.layer1(x) #卷积层1
        x = self.layer2(x) #卷积层2
        x = self.layer3(x) #卷积层3
        x = self.pool2(x) #池化
        x = x.view(x.size()[0], -1) #拉直 用于全连接层
        x = self.fc1(x) #全连接层
        x = self.relu2(x) #激活函数
        x = self.fc2(x) #全连接层
        return x 

def train_val(model, train_loader, val_loader, no_label_loader, device, epochs, optimizer, loss, thres, save_path): #训练和验证
    model = model.to(device) #将模型放到设备上
    #定义并初始化训练所需各个变量 用于全监督以及半监督学习共同使用
    semi_loader = None #初始化半监督学习 用于存储半监督数据集
    plt_train_loss = [] #定义数组用于存储训练损失
    plt_val_loss = [] #定义数组用于存储验证损失

    plt_train_acc = [] #定义变量用于存储训练准确率
    plt_val_acc = [] #定义变量用于存储验证准确率

    max_acc = 0.0 #初始化最大准确率

    for epoch in range(epochs): #遍历训练轮数
        #定义并初始化各个变量
        train_loss = 0.0 #训练损失值
        val_loss = 0.0 #验证损失值
        train_acc = 0.0 #训练准确率
        val_acc = 0.0 #验证准确率
        semi_loss = 0.0 #半监督损失值
        semi_acc = 0.0 #半监督准确率
        start_time = time.time() #用于计算训练时间
        #训练
        model.train() #设置模型为训练模式
        for batch_x, batch_y in train_loader: #遍历训练数据集
            x, target = batch_x.to(device), batch_y.to(device) #将数据和标签放到设备上训练
            pred = model(x) #计算预测值
            train_bat_loss = loss(pred, target) #计算预测值和真实值之间的损失
            train_bat_loss.backward() #梯度回传
            optimizer.step()  # 更新参数 之后要梯度清零否则会累积梯度
            optimizer.zero_grad() #梯度清零
            train_loss += train_bat_loss.cpu().item() #累加计算训练损失值
            train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy()) #累加计算训练准确率 并找到预测值最大的索引
        plt_train_loss.append(train_loss / train_loader.__len__()) #记录平均训练损失值
        plt_train_acc.append(train_acc/train_loader.dataset.__len__()) #记录平均训练准确率，
#以上为有标签的数据集的训练
#下面是对半监督模式的数据集是否为空进行检测，如果不为空那就对半监督模式的数据集进行训练
        if semi_loader!= None:#如果半监督训练集不为空
            for batch_x, batch_y in semi_loader: #从那个半监督学习数据集中按批为单位取出数据
                x, target = batch_x.to(device), batch_y.to(device) #将数据和标签放到设备上训练
                pred = model(x) #计算预测值
                semi_bat_loss = loss(pred, target) #计算半监督模式下预测值和真实值之间的损失
                semi_bat_loss.backward() #对半监督模式下的损失进行梯度回传
                optimizer.step()  # 更新参数 之后要梯度清零否则会累积梯度
                optimizer.zero_grad() #进行梯度清零
                semi_loss += train_bat_loss.cpu().item() #累加计算半监督模式下的损失值
                semi_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy()) #累加计算半监督模式下的准确率
            print("半监督数据集的训练准确率为", semi_acc/train_loader.dataset.__len__()) #打印输出
        
        #以下为验证部分
        model.eval() #设置模型为验证模式
        with torch.no_grad(): #取消梯度计算 用于验证集
            for batch_x, batch_y in val_loader: #按批次取出验证数据集计算准确率，提高效率
                x, target = batch_x.to(device), batch_y.to(device) #将数据和标签放到设备上验证
                pred = model(x) #计算预测值
                val_bat_loss = loss(pred, target) #计算预测值与真实值之间的损失
                val_loss += val_bat_loss.cpu().item() #累加计算验证损失值
                val_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy()) #将张量转化为numpy数组，可以直接用于数值运算 numpy只能在cpu上运行  比较预测值和真实值的类别 结果为一个布尔值的数组 通过np.sum()函数将布尔值转为整数值0或者1  最终得到的是每轮批次中预测正确的样本个数
        plt_val_loss.append(val_loss / val_loader.dataset.__len__()) #将验证集上的平均损失值存入数组 以便后续绘制损失曲线
        plt_val_acc.append(val_acc / val_loader.dataset.__len__()) #将验证集上的平均准确率存入数组 以便后续绘制准确率曲线

        if epoch%3 == 0 and plt_val_acc[-1] > 0.6: #设置准确率大于0.6时才可以把半监督的数据集放入训练集中训练
            semi_loader = get_semi_loader(no_label_loader, model, device, thres) #定义对象用于存储半监督数据集

        if val_acc > max_acc: #如果验证的准确率大于最大准确率
            torch.save(model, save_path) #设置模型保存路径
            max_acc = val_loss #更新最大准确率

        print('[%03d/%03d] %2.2f sec(s) TrainLoss : %.6f | valLoss: %.6f Trainacc : %.6f | valacc: %.6f' % \
              (epoch, epochs, time.time() - start_time, plt_train_loss[-1], plt_val_loss[-1], plt_train_acc[-1], plt_val_acc[-1])
              )  # 打印训练结果。 注意python语法， %2.2f 表示小数位为2的浮点数， 后面可以对应。

    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title("loss")
    plt.legend(["train", "val"])
    plt.show()


    plt.plot(plt_train_acc)
    plt.plot(plt_val_acc)
    plt.title("acc")
    plt.legend(["train", "val"])
    plt.show()

train_path = r"C:\Users\24494\Desktop\11\第四五节_分类代码\food_classification\food-11_sample\training\labeled" #设置训练集数据的路径
val_path = r"C:\Users\24494\Desktop\11\第四五节_分类代码\food_classification\food-11_sample\validation" #设置验证集数据的路径
no_label_path = r"C:\Users\24494\Desktop\11\第四五节_分类代码\food_classification\food-11_sample\training\unlabeled\00" #设置无标签数据的路径 用于半监督学习

train_set = food_Dataset(train_path, "train") #定义训练数据集对象 调用对应的变量 将数据对应的文件的路径导入 加载数据
val_set = food_Dataset(val_path, "val") #定义验证数据集对象 调用对应的变量 将对应数据的文件路径导入 加载数据
no_label_set = food_Dataset(no_label_path, "semi") #定义无标签数据集对象 调用对应的变量 将对应的数据文件路径导入 加载数据

#超参数 也就是使用到字典的方式来存储参数
lr = 0.001 #设置学习率
loss = nn.CrossEntropyLoss() #设置损失函数（直接调用pytorch中的交叉熵损失函数）
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4) #设置优化器（直接调用pytorch中的AdamW优化器， 传入参数: 模型参数， 学习率， 权重衰减）
device = "cuda" if torch.cuda.is_available() else "cpu"
save_path = "model_save/best_model.pth" 
epochs = 15 #设置训练轮数
thres = 0.99 #设置阈值

#dataloader可以将数据集按批次加载  参数中train_set设置数据集对象也就是上面foo_Dataset定义的对象，直接调用food_Dataset方法 batch_size设置批次大小 shuffle可以将数据打乱提高模型的泛化能力
train_loader = DataLoader(train_set, batch_size=16, shuffle=True) #训练集的dataloader
val_loader = DataLoader(val_set, batch_size=16, shuffle=True) #验证集的dataloader
no_label_loader = DataLoader(no_label_set, batch_size=16, shuffle=False) #无标签数据集的dataloader  用于半监督学习，其中模型通过无标签数据进行自我训练以及伪标签生成

# model = myModel(11) #这是自己定义的模型
model, _ = initialize_model("vgg", 11, use_pretrained=True)#调用initialize_model来初始化预定义模型 并且此时使用的是vgg模型 11表示输出层类别数 ，通常用于分类任务 use_pretrained表示是否使用预训练模型的权重

#完成半监督学习
train_val(model, train_loader, val_loader, no_label_loader, device, epochs, optimizer, loss, thres, save_path)#直接将模型，训练集，验证集，无标签数据集，设备，训练轮数，优化器，损失函数，阈值，保存路径一体化传入train_val函数 通过调用train_val函数完成半监督学习整个过程

#################从底层到顶层 从上至下 定义类 函数 变量 一层一层调用
#过程:
#数据准备:
    #1.train_loader 提供有标签的数据供模型进行训练
    #2.val_loader 提供有标签的数据供模型进行验证，帮助评估模型的性能并且防止过拟合
    #3.no_label_loader 提供无标签数据 用于半监督学习，通常结合模型的预测生成为标签，然后在训练中使用这些伪标签来增强学习
#模型初始化:
    #4.initialize_model 初始化预定义VGG模型，为以下的train_val函数提供模型
#####训练和验证过程:
    #5.train_val综合所有准备好的数据，模型，设备，超参数，优化器，损失函数，阈值，保存路径，顺序完成训练和验证过程