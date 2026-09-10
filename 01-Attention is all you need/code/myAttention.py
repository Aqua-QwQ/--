from torch import nn
import torch

### 注意力点积
def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    # 1. 计算得分，并缩放
    scores = (q @ k.transpose(-2,-1))/(d_k**0.5)
    # 2. 在 softmax 前屏蔽禁止位置
    if mask is not None:
        scores = scores.masked_fill(mask,float("-inf"))
    # 3. 沿 key 维度计算权重
    weights = torch.softmax(scores,dim=-1)
    # 4. 对 V 加权求和
    output = weights @ v
    return output, weights

### 自注意力
class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        output, weights = scaled_dot_product_attention(q, k, v, mask)
        return output, weights

### 拆头函数
def split_heads(x, num_heads):
    # 输入：[B, L, D]
    B, L, D = x.shape
    assert D % num_heads == 0
    x = x.reshape(B, L, num_heads, D//num_heads)
    x = x.transpose(1, 2)
    return x   # 输出：[B, H, L, Dh]

### 合头函数
def merge_heads(x):
    # 输入：[B, H, L, Dh]
    B, H, L, Dh = x.shape
    x = x.transpose(1, 2)
    x = x.reshape(B, L, H*Dh)
    return x  # 输出：[B, L, H * Dh]

### 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        # 投影后拆头
        q = split_heads(self.q_proj(query), self.num_heads)
        k = split_heads(self.k_proj(key),self.num_heads)
        v = split_heads(self.v_proj(value),self.num_heads)
        # 每个头分别计算注意力
        head_output, weights = scaled_dot_product_attention(
            q, k, v, mask
        )
        # 合头，再做输出投影
        output = merge_heads(head_output)
        output = self.out_proj(output)
        return output, weights