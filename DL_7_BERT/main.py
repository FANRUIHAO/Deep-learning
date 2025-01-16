import random
import torch
import torch.nn as nn
import numpy as np
import os

from model_utils.data import get_data_loader
from model_utils.model import myBertModel
from model_utils.train import train_val


def seed_everything(seed):#设置随机种子 确保代码的可重复性和结果的可复现性  是的结果更加稳定，便于调试
    torch.manual_seed(seed) # 为CPU设置随机种子
    torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # 为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False #控制卷积算法 用false保证稳定性
    torch.backends.cudnn.deterministic = True#保证结果的确定性
    random.seed(seed)#打乱数据
    np.random.seed(seed)#用于大规模数据处理
    os.environ['PYTHONHASHSEED'] = str(seed)#设置哈希种子，它控制哈希函数的随机性，确保在每次运行时相同的输入值能得到相同的哈希值

seed_everything(0)


#超参数
lr=0.0001
batchsize= 4
loss = nn.CrossEntropyLoss()
bert_path = 'bert-base-uncased'
num_class = 2
data_path = 'data/'#数据所在的文件的路径
max_acc = 0.6#要求准确率
val_epoch = 1#验证的频率

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#判断是否有cuda可用
model = myBertModel(bert_path, num_class)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, wegtht_decay=0.01)#优化器

train_loader, val_loader = get_data_loader(data_path, batchsize)#数据加载器

epochs = 5
save_path = 'model.pth'

schedular = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=le-9)#学习率调整器

para = {
    "model": model,
    "train_loader": train_loader,
    "val_loader": val_loader,
    "scheduler": schedular,
    "optimizer": optimizer,
    "loss": loss,
    "epochs": epochs,
    "save_path": save_path,
    "device": device,
    "max_acc":max_acc,
    "val_epoch": val_epoch,
}