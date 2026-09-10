import torch
from myEncoder import *

layer = EncoderLayer(d_model=12, num_heads=3, d_ff=48)
layer.eval()  # 关闭 dropout，便于比较

x = torch.randn(2, 5, 12)

# [B, 1, 1, L]：广播到所有头和 query
# 假设每条序列的最后一个位置是 PAD
mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
mask[:, :, :, -1] = True

x_changed = x.clone()
x_changed[:, -1, :] += 100

with torch.no_grad():
    y = layer(x, mask)
    y_changed = layer(x_changed, mask)

print("输出形状:", y.shape)
print("有效位置是否不变:",
      torch.allclose(y[:, :-1, :], y_changed[:, :-1, :]))