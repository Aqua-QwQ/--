# 步骤 00：环境确认与第一次自动求导

## 当前检查结果

检查日期：2026-09-05。

- 当前机器架构：arm64。
- Conda 安装位置：`/Users/aquaqwq/miniconda3`。
- 当前默认环境：base，Python 3.13.12，未检测到 PyTorch。
- 已有 cs336 环境：Python 3.14.4、PyTorch 2.11.0，已验证可以导入并在 CPU 上计算和求导。
- 当前助手执行进程中：MPS built=True、MPS available=False、CUDA available=False。
- 此结果只说明当前执行进程未能使用 MPS，不据此认定 Mac 不支持 GPU 加速；需要在用户自己的终端复查。
- 用户终端复查结果：cs336 解释器路径正确，PyTorch 2.11.0，MPS available=True；已成功计算 y=14、梯度 `[2., 4., 6.]`。

先复用已有 cs336 环境进行入门练习，不修改其中依赖。独立项目环境及 environment.yml 后续再确定，此步骤尚未全部完成。

## 学习目标

- 理解 Conda 环境决定使用哪个 Python 和哪些包。
- 确认终端运行的是预期解释器。
- 理解张量、requires_grad、backward 和 grad 的基本关系。

## 终端操作

```bash
conda activate cs336
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

随后输入 `python` 进入交互解释器，逐行运行：

```python
import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = (x ** 2).sum()
print(y)

y.backward()
print(x.grad)
```

预期：y 的值为 14，x.grad 为 `[2., 4., 6.]`。显示中可能带有描述计算图的额外信息。

这里 y = x₁² + x₂² + x₃²，因此各分量的导数是 2xᵢ。`backward()` 计算梯度，结果存放在 `x.grad`，不会自动更新 x。

## 已验证与待完成

- [x] 助手进程成功导入已有环境的 PyTorch。
- [x] 助手进程验证 CPU 计算得到 14，梯度得到 `[2.0, 4.0, 6.0]`。
- [x] 学习者在自己的终端激活环境并复现。
- [x] 学习者正确预测 `(3 * x).sum()` 的梯度为 `[3., 3., 3.]`。
- [ ] 学习者解释 backward 是否修改 x，并理解梯度累积。
- [x] 在用户终端复查 MPS 可用性：True。
- [ ] 确定独立项目环境，补充 environment.yml 和 README 运行说明。

## 首次练习中的问题

第一次把 Python 代码直接粘贴到了 zsh 中，出现 `command not found: import` 等错误；输入 `python`，看到 `>>>` 提示符后重试成功。

- `%` 是本机 shell 提示符，适合运行 `conda activate`、`python` 等命令。
- `>>>` 是 Python 交互提示符，适合运行 `import torch` 等 Python 代码。
- `quit()` 或 Ctrl-D 可以退出 Python 解释器。

下一项小练习：观察 `backward()` 不修改 x，并验证多次反向传播时叶子张量的梯度会累积；每次反向前重新计算 y，避免重复使用已释放的计算图。
