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

海森矩阵判断是local minima还是 saddle point  
    local minima在别的情况下可能是saddle point

batch：大的batch反而耗时短，但是小的batch却可以达到更好的结果

-----------------------------------
learning rate scheduling

learning rate：如果在某一方向变化很小的话可以调大尝试
如果在某一方向变化很陡峭的话可以尝试调小

学习率可以动态调整的：RMSProp方法   
    Adam：RMSProp + Momentum

warm up（黑科技）：可用于bert训练
    它会使得训练开始的时候学习率为0，01甚至更小，后期在慢慢调大比如0.1

meta learning（学习如何学习）：？？？？？？？？？？？？？？、

其实如果正确率总是100的话，反而是不利于判断generator是否在进步


有时候generator生成的图片很棒，但并不代表是有效的，可能和数据集是一模一样的，就算通过相似度来判断，也有可能只是反转的图片  

generator不仅可以文字生成图片，也可以图片生成图片 pix2pix  
gan+supervised 效果是最好的  


Transformer：
    Encode：先输入一排向量 通过encoder的许多层Block，最终得到block sequence    

    transformer特有机制：通过self-attention得到新的vector后加上输入得到residual，进而通过Layer norm
                        对输入的向量肌酸mean和Standard deviation然后输出另外一个向量，并且不需要考虑batch norm 
                        
                        进一步的通过全线性层，输出加上输入
self-attention ----> masked self-attention  加入masked可以增加模型鲁棒性
    ATdecoder只能一个接一个串联着训练，而NATdecoder可以并行将所有输入一口气处理，提高效率  
--------------------------------------------------------------  
Guided Attention 可以用于（语音辨识/合成）中使得模型能够听到之前没听到的语音内容
    就是因为模型自己识别向量时，因为没有很好的识别顺序，导致有的语音向量没有识别到；而guided attention可以引导模型去有序的辨识语音向量  


Beam Search 相当于二叉树，最大生成路径  找出的最大生成路径就叫做greedy decoding
    用比较有效的方法，找一个不是特别精准的路径 叫做Beam search（有时候有用，有时候废物）
    在需要创造的情况下，往往答案不是唯一的，那么beam search就有可能没用；在答案确定明了的情况下，beam search比较有用  

语音合成的时候一般可以用到TTS

aattention中可以使用clustering来计算query和key，将一样的cluster进行计算，不一样的直接设为0
要不要计算attention？那些计算要不要计算attention，使用sinkhorn sorting network在matrix上对cluster进行计算

qkv相乘，三者相乘的顺序不一样，虽然结果是一样的，但是运算量是不一样的


seq2seq：其实就是输入一串句子生成另一串句子  
    encoder和decoder都是rnn  
    将编码器最后时间隐状态来出初始decoder隐状态来完成信息传递  
    经常用BLEU衡量生成序列的好坏  




















