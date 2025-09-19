import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(TransformerEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x):
        x = self.embedding(x) * np.sqrt(d_model)
        x = self.positional_encoding(x)
        return x

# 示例参数
vocab_size = 10000
d_model = 512
seq_length = 10

# 创建输入嵌入和输出嵌入实例
input_embedding = TransformerEmbedding(vocab_size, d_model)
output_embedding = TransformerEmbedding(vocab_size, d_model)

# 示例输入
input_seq = torch.randint(0, vocab_size, (seq_length, 32))  # (seq_length, batch_size)
output_seq = torch.randint(0, vocab_size, (seq_length, 32))  # (seq_length, batch_size)

# 获取嵌入表示
input_embedded = input_embedding(input_seq)
output_embedded = output_embedding(output_seq)

print("输入嵌入形状:", input_embedded.shape)
print("输出嵌入形状:", output_embedded.shape)