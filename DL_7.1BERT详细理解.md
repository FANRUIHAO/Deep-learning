BERT实质上就是多层encoder堆叠在一起（不是多层transformer堆叠在一起
NSP（next sentence prediction） 任务是判断两个句子之间的关系
    需要符号来标注不同两个句子（cls向量不能代表整个句子的语义信息）

transformer中用的是正余弦函数
BERT用的是随机初始化然后让模型自己学出来

预训练：
    在一般预训练中，实际上数据是没有标签的，若以要用无监督学习

    模型：
        MLM 掩码语言模型（AR）：优化目标顺序为串行
        无监督目标函数（AE）：并行计算，互相不受影响
                利用mask模型