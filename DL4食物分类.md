import random
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.mamual_seed(seed)
    torch.cuda.mamual_seed_all(seed)
    torch.backends.cunn.benchmark = False
    torch.backends.cunn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ[]

a = random.randint(1, 5)

print(a)



### 数据增广

# Adam 和 Adamw
Adam 更稳定 效果更好，很难出现梯度爆炸
1、梯度的改变
    sgd即 只看当前点的梯度 

    Adam 要看前面点的梯度 再看当前点的梯度  （即综合以前和当前点的梯度）
2、学习率的改变

Adamw在Adam基础上加上权重衰减 weight__decay

## 注意：准确率

train_acc += np.sum(np.argmax(pred.detach().cpu().numpy(), axis=1) == target.cpu().numpy())

np.argmax(pred.detach().cpu().numpy(), axis=1) 用概率表示数据  data1就是target 
argsoft作用是排序从小到大的值的下标
axis轴为1就是横着取值  若为0就是竖着取值

target.cpu().numpy() 即等式两侧各个对应的元素是否相等，不相等就是false=0，相等就是true=1  然后把所有的判断元素结果相加

最难的是数据处理（根据输入的路径将x，y读出来）
要保持维度，使得计算顺利进行