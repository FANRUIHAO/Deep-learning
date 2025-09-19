### 常见输入输出

输入一段，输出一段  Seq2 seq
    输出长度由模型自行决定
    本质上是序列到序列的任务，采用的架构就是transformer架构

transformer架构：与rnn区别在于不需要等待前一个字生成后才能生成下一个字，可以并行生成，效率更高
    结构：
    1、encoder模型（bert就属于encoder）
    2、decoder模型:接收一个start token，然后开始一个字一个字的推理。一次只生成一个字，通过softmax生成字的概率分布，同时把生成的字作为输入，再生成下一个字。

        用label(标签)作为decoder的输入可以实现并行的效果。
        mask self-attention
        把当前输入的字后面所有的内容都用mask，使其看不到后面的字，而后面的字自然也皆输出0

    如何知道要输出多长？
        用end来解决，即在输出最后一个字的后面加上一个end，否则会不断循环输出下去

    一个生成任务如何训练？
        求cross entropy loss
    输入的内容称作Ground Truth，其实就是标签

#### 平常任务：
            经过模型 提取特征，经过分类头，得到结果

#### 生成任务：
            ·训练的时候有mask，但是标签是直接作为输入的
            ·测试的时候只能一个一个字的来，整体串行，无法并行

            Beam serch：规定每次要看的步数

                        在推测的时候选择贪婪策略还是大局观策略
                        选择两个概率不同的选项时，选择概率高的那个
                    ·贪婪：每步只看当前最好的
                    ·大局：整体来看，而不是局限于某一两个对象

得到预测值：
    k v a'都是由encoder得出， q由self-attention(mask)得到



思考：
encoder decoder输入是什么
为社么要mask
cross attention是如何cross的
训练的loss是如何计算的
    