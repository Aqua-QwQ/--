import torch
from myEmbedding import *

###
module = TransformerEmbedding(vocab_size=10, d_model=4)
module.eval()  # 暂时关闭 dropout，便于观察

token_ids = torch.tensor([[2, 2, 5]])
output = module(token_ids)

print(output.shape)  # [1, 3, 4]

module = TransformerEmbedding(vocab_size=10, d_model=4)
module.eval()

token_ids = torch.tensor([[2, 2, 5]])
output = module(token_ids)

print("输出形状:", output.shape)
print("位置 0:", output[0, 0])
print("位置 1:", output[0, 1])