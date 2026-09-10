import torch
from torch import *
from myEmbedding import *
from myAttention import *
from myFFN import *

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力：Q、K、V 的输入都来自 x
        attn_output, _ = self.attention(x, x, x, mask=mask)
        # 第一次残差连接和归一化
        x = self.norm1(x + self.dropout1(attn_output))
        # 第二次残差连接和归一化
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff,
                 num_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x