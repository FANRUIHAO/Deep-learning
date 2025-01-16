#data负责产生dataloader
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split #给X，Y和分割比率， 分割出来一个训练集和测试集的X和Y
import torch


def read_file(path):

    with open(path, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i== 0:
                continue
            line = line.strip("\n")#strip()作用为去掉句子中的换行
            line = line.split(",", 1)#split()作用为将句子按照逗号分割，1表示分割次数
            data.append(line[1])
            label.append(line[0])
    print("读了%d行数据"%len(data))
    return data, label

file = "../xxxx"#表示数据集所在文件的路径
read_file(file)#读取数据

class jdDataset(Dataset):
    def __init__(self, data, label):
        self.X = data
        self.Y = torch.LongTensor([int(i) for i in label])
    def __getitem__(self, index):
        return self.X[item], self.Y[item]
    
    def __len__(self):
        return len(self.Y)
    



def get_data_loader(path, batchsize, val_size=0.2): #读入数据，分割数据
    data, label = read_file(path)
    train_x, val_x, train_y, val_y = train_test_split(data, label, test_size=val_size, shuffle=True, stratify=label)
    train_set = jdDataset(train_x, train_y)
    val_set = jdDataset(val_x, val_y)
    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchsize, shuffle=False)
    return train_loader, val_loader