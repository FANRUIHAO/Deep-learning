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