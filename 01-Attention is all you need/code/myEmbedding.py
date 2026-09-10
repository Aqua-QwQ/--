from torch import nn
import torch

### 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        assert d_model % 2 == 0  # 当前版本先支持偶数维度

        positions = torch.arange(max_len).float().unsqueeze(1)
        even_dims = torch.arange(0, d_model, 2).float()
        angles = positions / (10000 ** (even_dims / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, D]
        length = x.size(1)
        if length > self.pe.size(0):
            raise ValueError("输入序列超过最大支持长度")

        return x + self.pe[:length]

### Embedding实现
class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        # 1. 查找词向量，并乘以 sqrt(d_model)
        x = self.embedding(token_ids)*(self.d_model** 0.5)
        # 2. 加位置编码
        x = self.position(x)
        # 3. 应用 dropout
        x = self.dropout(x)
        return x