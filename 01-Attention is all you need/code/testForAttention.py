import torch
from myAttention import *

###
def test1():
    torch.manual_seed(42)

    q = torch.randn(2, 3, 8)
    k = torch.randn(2, 3, 8)
    v = torch.randn(2, 3, 4)

    mask = torch.triu(
        torch.ones(3, 3, dtype=torch.bool),
        diagonal=1,
    )

    output, weights = scaled_dot_product_attention(q, k, v, mask)

    print("输出形状:", output.shape)
    print("第一个样本的权重:", weights[0])
    print("每行权重之和:", weights.sum(dim=-1))
    v_changed = v.clone()
    v_changed[:, 2, :] += 100  # 只修改最后一个位置的 value

    output_changed, _ = scaled_dot_product_attention(
        q, k, v_changed, mask
    )

    print("前两个位置是否不变:",
        torch.allclose(output[:, :2, :], output_changed[:, :2, :]))

    print("最后一个位置是否不变:",
        torch.allclose(output[:, 2, :], output_changed[:, 2, :]))

#test1()

###
def printTest():
    print("-"*40)

###
def test2():
    model = SelfAttention(d_model=8)
    x = torch.randn(2, 3, 8)

    output, weights = model(x)

    # 临时构造一个标量目标，只检查梯度是否能传回投影层
    loss = output.square().mean()
    loss.backward()

    print("输出形状:", output.shape)
    print("Q 投影的梯度形状:", model.q_proj.weight.grad.shape)

#test2()

###
def test3():
    x = torch.arange(2 * 5 * 12).reshape(2, 5, 12)

    heads = split_heads(x, num_heads=3)
    restored = merge_heads(heads)

    print(heads.shape)                  # [2, 3, 5, 4]
    print(torch.equal(x, restored))     # True

#test3()

###
def test4():
    model = MultiHeadAttention(d_model=12, num_heads=3)
    x = torch.randn(2, 5, 12)

    mask = torch.triu(
        torch.ones(5, 5, dtype=torch.bool),
        diagonal=1,
    )

    output, weights = model(x, x, x, mask)

    print(output.shape)   # [2, 5, 12]
    print(weights.shape)  # [2, 3, 5, 5]

#test4()

###
def test5():
    model = MultiHeadAttention(d_model=12, num_heads=3)

    query = torch.randn(2, 3, 12)
    memory = torch.randn(2, 5, 12)

    output, weights = model(query, memory, memory)

    print("输出形状:", output.shape)
    print("权重形状:", weights.shape)

    # 检查梯度能否传回各个投影层
    loss = output.square().mean()
    loss.backward()

    for name, parameter in model.named_parameters():
        print(name, parameter.grad is not None)

test5()