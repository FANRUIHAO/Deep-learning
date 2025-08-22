batch就是把数据按组划分，1 epoch就是将所有的组分别训练一遍

neuron 神经元

训练思路：  
![img_8.png](img_8.png)  

处理过拟合的方法：  
1、减少神经元数量  
2、减少特征  
3、dropout  

对于训练效果好坏的判断：尽量不要参照private testing set  
参照training set以及 validation set最好  

n折交叉验证：  
![img_9.png](img_9.png)  

训练集和测试集分布不同的话可能会导致mismatch