import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_sizedfdfd
        self.heads = heads
        # 每个注意力头的维度
        self.head_dim = embed_size // heads

        # 确保特征维度可以被头数整除
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        # 定义 Q, K, V 的线性映射层
        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        
        # 定义输出的线性映射层
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0] # 获取 Batch size
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 1. 经过线性层映射，得到 Q, K, V
        v = self.W_v(values)
        k = self.W_k(keys)
        q = self.W_q(query)

        # 2. 划分多头注意力: 
        # 将形状由 (Batch, SeqLen, EmbedSize) 变换为 (Batch, SeqLen, Heads, HeadDim)
        v = v.view(N, value_len, self.heads, self.head_dim)
        k = k.view(N, key_len, self.heads, self.head_dim)
        q = q.view(N, query_len, self.heads, self.head_dim)

        # 3. 交换维度以便于矩阵乘法计算:
        # 变为 (Batch, Heads, SeqLen, HeadDim)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)

        # 4. 计算注意力分数: Q * K^T / sqrt(d_k)
        # q 的形状: (Batch, Heads, QueryLen, HeadDim)
        # k 经过转置后形状: (Batch, Heads, HeadDim, KeyLen)
        # energy 的形状: (Batch, Heads, QueryLen, KeyLen)
        energy = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 如果有 mask 掩码（例如在 Decoder 中防止看到未来的词），则把 mask 区域填充为极小值
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # 对最后一个维度做 softmax，得到权重归一化的注意力分布
        attention = torch.softmax(energy, dim=-1)

        # 5. 注意力权重乘以 V
        # attention 的形状: (Batch, Heads, QueryLen, KeyLen)
        # v 的形状: (Batch, Heads, ValueLen, HeadDim) (这里的 ValueLen = KeyLen)
        # out 的形状: (Batch, Heads, QueryLen, HeadDim)
        out = torch.matmul(attention, v)

        # 6. 将多个头的结果拼接回去
        # (Batch, Heads, QueryLen, HeadDim) -> (Batch, QueryLen, Heads, HeadDim) -> (Batch, QueryLen, EmbedSize)
        out = out.transpose(1, 2).contiguous().view(N, query_len, self.embed_size)

        # 7. 经过最后一次线性层输出
        out = self.fc_out(out)

        return out

if __name__ == "__main__":
    # 简单的测试用例
    BATCH_SIZE = 2
    SEQ_LENGTH = 5     # 序列长度 (例如句子中有 5 个单词)
    EMBED_SIZE = 256   # 词向量的特征维度
    HEADS = 8          # 注意力头数

    # 实例化多头注意力机制
    mha = MultiHeadAttention(embed_size=EMBED_SIZE, heads=HEADS)

    # 随机生成一些测试数据，模拟 self-attention (自注意力) 下的 Q=K=V
    # 形状为 (Batch, Sequence Length, Embedding Size)
    x = torch.randn((BATCH_SIZE, SEQ_LENGTH, EMBED_SIZE))

    # 前向传播
    output = mha(values=x, keys=x, query=x)

    print(f"输入特征维度: {x.shape}")
    print(f"经过多头注意力机制后的输出维度: {output.shape}")
