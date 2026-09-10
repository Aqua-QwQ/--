# Transformer 学习与实现计划

更新日期：2026-09-05

## 目标与推进方式

通过实现一个小型 Encoder–Decoder Transformer，理解注意力、张量变换、训练和自回归生成。按步骤推进，不按天数划分；每一步完成交付与验收后再继续。

当前基础：掌握 Python 基本语法，PyTorch 不熟悉；有 Mac M5、RTX 4060 笔记本和 Conda，计划投入一周学习。优先完成步骤 00–10 的复制任务闭环；步骤 11–12 是后续进阶，不要求在一周内全部完成。

采用学习型复现：缩小模型和数据规模，记录与论文的差异。复制任务用于验证实现，不代表完成了论文的翻译实验或指标复现。

协作方式：先讨论当前步骤的原理和接口，再由学习者尝试实现；助手检查实现与结果，必要时提供局部参考代码。

## 文件组织与约定

以下所有路径相对于 `code/`。这是待实现的目录规划，本次仅更新 `plan.md`，其余文件按步骤创建。

```text
code/
├── plan.md
├── README.md                         # 项目入口与运行命令
├── environment.yml                   # Conda 基础环境
├── docs/                             # 按步骤记录理解与验收结果
│   ├── 00_environment.md
│   ├── 01_pytorch_basics.md
│   ├── 02_data_and_shapes.md
│   ├── 03_scaled_dot_product_attention.md
│   ├── 04_multi_head_attention.md
│   ├── 05_embedding_and_position.md
│   ├── 06_encoder.md
│   ├── 07_decoder.md
│   ├── 08_transformer.md
│   ├── 09_training.md
│   ├── 10_generation_and_evaluation.md
│   ├── 11_translation.md              # 进阶
│   └── 12_experiments_and_summary.md   # 进阶
├── exercises/
│   └── pytorch_basics.py
├── transformer/
│   ├── __init__.py
│   ├── masks.py
│   ├── attention.py
│   ├── embeddings.py
│   ├── layers.py
│   ├── model.py
│   └── decoding.py
├── data/
│   ├── copy_task.py
│   └── translation.py                # 进阶
├── configs/
│   ├── copy.json
│   └── translation.json              # 进阶
├── checks/                           # 小型、可直接运行的验证脚本
│   ├── check_data.py
│   ├── check_attention.py
│   ├── check_multi_head.py
│   ├── check_embeddings.py
│   ├── check_encoder.py
│   ├── check_decoder.py
│   └── check_model.py
├── train.py
├── evaluate.py
└── outputs/                          # 实际运行后生成，不能提前视为完成
    ├── copy/
    │   ├── config.json
    │   ├── metrics.csv
    │   ├── best.pt
    │   ├── predictions.jsonl
    │   └── evaluation.json
    ├── translation/                  # 进阶
    └── ablations/                    # 进阶
```

统一约定：

- 使用 PyTorch 自动求导及 `Linear`、`Embedding`、`LayerNorm`、`Dropout`、损失函数与优化器；自己实现注意力和模型组织，不调用 `nn.Transformer` 或封装好的多头注意力。
- `B` 表示 batch 大小，`S` 表示源序列长度，`T` 表示目标序列长度，`D` 表示模型维度，`H` 表示头数，`Dh = D / H`。
- token ID 的形状为 `[B, L]`，隐藏表示为 `[B, L, D]`；注意力分数的最后两维是 query 长度与 key 长度。
- 布尔 mask 统一约定 `True` 表示禁止关注。显式处理广播，不依赖不同 API 的默认语义。
- 每份步骤文档包含：学习问题、相关公式或数据流、输入输出形状、实现选择、验证命令与实际结果、尚未解决的问题。
- 验证脚本只检查关键正确性，例如 mask 泄漏、目标移位和梯度传播，不追求复杂测试框架。

## 步骤 00：准备可运行的环境

**文件**：`environment.yml`、`README.md`、`docs/00_environment.md`。

**具体实现**：

1. 确认主开发机、操作系统、Conda 和 Python 环境；建立独立环境。
2. 安装适合该设备的 PyTorch，记录实际版本与安装命令。
3. 验证 CPU 张量计算和自动求导；检查 CUDA 或 MPS 是否可用。
4. 确定后续设备选择方式，保留 CPU 调试入口。
5. 在 README 中写明从环境创建到运行验证脚本的命令。

**交付物**：可复建的环境说明、设备检测结果、一次张量求导的实际输出。

**验收**：新终端激活环境后能导入 PyTorch，完成张量计算和 `backward()`；加速设备不可用时仍能在 CPU 继续学习。

## 步骤 01：掌握最小 PyTorch 训练流程

**文件**：`exercises/pytorch_basics.py`、`docs/01_pytorch_basics.md`。

**具体实现**：

1. 练习张量创建、索引、广播、`matmul`、`reshape`、`transpose` 和 `softmax`。
2. 用可学习的简单映射生成合成数据，例如线性回归。
3. 用 `nn.Module` 定义小网络，实现前向计算。
4. 写出 `zero_grad → forward → loss → backward → step` 循环。
5. 检查参数梯度与更新前后的参数，记录 loss 的变化。

**交付物**：一个可直接运行的基础练习脚本；一份解释训练循环和张量操作的笔记。

**验收**：loss 明显下降，参数发生更新；能解释梯度为什么需要清零，以及 `backward()` 和 `step()` 的区别。

## 步骤 02：定义复制任务、特殊符号与 mask

**文件**：`data/copy_task.py`、`transformer/masks.py`、`checks/check_data.py`、`docs/02_data_and_shapes.md`。

**具体实现**：

1. 定义互不重叠的 `PAD`、`BOS`、`EOS` ID 和普通 token 范围。
2. 生成变长整数序列，建立 Dataset、DataLoader 与 padding 的批处理函数。
3. 约定一个样本：源序列为 `[a, b, c, EOS]`，解码器输入为 `[BOS, a, b, c]`，标签为 `[a, b, c, EOS]`；补齐时使用 PAD。
4. 实现 key padding mask `[B, 1, 1, Lk]` 和因果 mask `[1, 1, T, T]`。
5. 说明解码器自注意力如何合并目标 padding 与因果 mask；交叉注意力如何使用源 padding mask。
6. 固定数据种子，建立独立训练集、验证集和测试集，检查完整序列没有跨集合重复。

**交付物**：数据生成与批处理代码、mask 函数、一个打印样本及形状的验证脚本、目标移位示意图。

**验收**：变长序列正确补齐；标签与解码器输入错开一位；未来位置与源 PAD 被正确屏蔽；普通样本不产生所有 key 均被屏蔽的 query 行。

## 步骤 03：实现缩放点积注意力

**文件**：`transformer/attention.py`、`checks/check_attention.py`、`docs/03_scaled_dot_product_attention.md`。

**具体实现**：

1. 实现 `scaled_dot_product_attention(q, k, v, mask=None, dropout=None)`。
2. 计算 `QKᵀ / sqrt(Dh)`，应用 mask，再沿 key 维做 softmax。
3. 对注意力权重应用可选 dropout，与 V 相乘，返回输出及用于检查的注意力权重。
4. 用极小张量展示每一步中间值，解释 Q、K、V 的作用。
5. 约定全屏蔽行的处理方式，避免悄悄产生 NaN。

**交付物**：注意力函数、可核对的小张量计算示例、mask 正确性检查。

**验收**：关闭 dropout 时，合法行的权重和为 1，被屏蔽位置权重为 0；改变被屏蔽位置的 V 不影响输出；输出与梯度有限。

## 步骤 04：实现多头注意力

**文件**：继续完善 `transformer/attention.py`；新增 `checks/check_multi_head.py`、`docs/04_multi_head_attention.md`。

**具体实现**：

1. 实现 `MultiHeadAttention(d_model, num_heads, dropout)`，检查 D 能被 H 整除。
2. 建立 Q、K、V 的线性投影和输出投影。
3. 完成 `[B, L, D] → [B, H, L, Dh]` 的变换、调用注意力、合并头。
4. 接口支持分别传入 query、key、value，以复用到自注意力和交叉注意力。
5. 测试源长度与目标长度不同的情况，避免误把所有长度视为相同。

**交付物**：多头注意力模块、拆头与合头的形状图、自注意力和交叉注意力示例。

**验收**：当 query 长度为 T、key/value 长度为 S 时，输出为 `[B, T, D]`，权重为 `[B, H, T, S]`；各投影参数能收到有限梯度。

## 步骤 05：实现词嵌入与位置编码

**文件**：`transformer/embeddings.py`、`checks/check_embeddings.py`、`docs/05_embedding_and_position.md`。

**具体实现**：

1. 实现 token embedding，并按设计加入 `sqrt(D)` 缩放。
2. 实现正弦、余弦位置编码，用 buffer 保存，不作为可训练参数。
3. 把词嵌入和位置编码相加，再应用 dropout。
4. 检查不同序列长度、设备迁移及最大长度边界。

**交付物**：嵌入与位置编码模块、若干位置的编码数值、解释位置编码作用的笔记。

**验收**：输出为 `[B, L, D]`；不同位置的编码不同；编码不参与梯度更新；模型迁移设备后仍能运行。

## 步骤 06：实现 Encoder

**文件**：`transformer/layers.py`、`checks/check_encoder.py`、`docs/06_encoder.md`。

**具体实现**：

1. 实现逐位置前馈网络：`Linear(D, Dff) → ReLU → Dropout → Linear(Dff, D)`。
2. 实现 EncoderLayer：自注意力、残差连接与归一化、前馈网络、残差连接与归一化。
3. 第一版明确使用 Post-LN，即子层输出经过 dropout 后与输入相加，再做 LayerNorm；在文档中画清顺序。
4. 堆叠 N 个独立参数的 EncoderLayer，传递源 padding mask。

**交付物**：前馈模块、EncoderLayer 和 Encoder、单层数据流图。

**验收**：输入输出保持 `[B, S, D]`；各层参数独立；前向与反向均正常；关闭 dropout 后修改被屏蔽源位置的表示，不改变有效位置的输出。

## 步骤 07：实现 Decoder

**文件**：继续完善 `transformer/layers.py`；新增 `checks/check_decoder.py`、`docs/07_decoder.md`。

**具体实现**：

1. 实现带因果 mask 的目标自注意力。
2. 实现交叉注意力：Q 来自解码器，K、V 来自编码器输出。
3. 加入前馈网络，为三个子层分别配置残差连接与归一化。
4. 堆叠 N 个独立参数的 DecoderLayer。
5. 明确区分目标自注意力 mask 与源 padding mask。

**交付物**：DecoderLayer 和 Decoder、三种注意力的信息来源对照、因果性验证脚本。

**验收**：关闭 dropout，固定编码器输出，改变目标序列未来 token 后，较早位置的解码器输出保持不变；S 与 T 不同时也能运行。

## 步骤 08：组装完整 Transformer

**文件**：`transformer/model.py`、`configs/copy.json`、`checks/check_model.py`、`docs/08_transformer.md`。

**具体实现**：

1. 组合源嵌入、目标嵌入、Encoder、Decoder 和词表输出投影。
2. 提供 `encode(src_ids)`、`decode(tgt_ids, memory, src_mask)` 和 `forward(src_ids, tgt_ids)` 等清晰接口。
3. `forward` 输出未经过 softmax 的 logits `[B, T, vocab_size]`，直接供交叉熵使用。
4. 集中配置词表、层数、D、H、Dff、dropout、最大长度和随机种子。
5. 初始可尝试 2 层、D=128、H=4、Dff=512 的小模型；这是起点，不是性能承诺。
6. 记录参数量、完整形状流，以及归一化、权重共享等实现选择。

**交付物**：完整模型、小模型配置、从 token 到 logits 的架构图、与论文差异的初始清单。

**验收**：一个真实批次能计算有限 loss 并反向传播；有效模块获得梯度；保存并重新加载权重后，在 eval 模式下输出一致。

## 步骤 09：实现训练并过拟合小批样本

**文件**：`train.py`、`docs/09_training.md`；更新 `configs/copy.json`、`README.md`。

**具体实现**：

1. 接入 DataLoader、模型、交叉熵和 Adam 优化器；用 `ignore_index=PAD` 排除补齐标签。
2. 分别实现训练与验证流程，正确切换 `train()`、`eval()` 与梯度开关。
3. 先固定一小批样本反复训练，确认模型能记住它，再扩展到完整复制任务训练集。
4. 按有效 token 数汇总 loss 和 token accuracy，不能让 PAD 提高准确率。
5. 记录训练步、训练/验证指标、学习率与运行配置，保存验证集表现最好的 checkpoint。
6. 第一版用简单训练配置；把尚未实现的论文学习率调度、label smoothing 等写入差异清单。

**交付物**：训练入口，以及实际运行生成的 `outputs/copy/config.json`、`metrics.csv`、`best.pt`；文档记录小批过拟合结果与正式训练结果。

**验收**：小批样本的非 PAD token accuracy 接近 100%，loss 显著下降；完整训练的验证指标有改善；checkpoint 可读取且包含恢复模型所需配置。训练准确率不能替代下一步的生成验收。

## 步骤 10：实现自回归生成与独立评估

**文件**：`transformer/decoding.py`、`evaluate.py`、`docs/10_generation_and_evaluation.md`；更新 `README.md`。

**具体实现**：

1. 实现贪心解码：源序列编码一次，从 BOS 开始，每轮取最后位置的预测并追加。
2. 生成只使用源输入和已生成的前缀；遇到 EOS 或最大生成长度时停止。
3. 先实现单样本生成，再处理批内不同样本提前结束的情况；暂不做 KV cache。
4. 加载 best checkpoint，在独立测试集上生成；测试集不用于选 checkpoint 或反复调参。
5. 计算生成 token accuracy 和整句完全匹配率：去掉 BOS/PAD，按首个 EOS 截断并保留 EOS，缺失或多余 token 计错；两端均缺失的补齐位置不计入分母。
6. 导出源序列、目标、预测与是否匹配，保留成功和失败样例。

**交付物**：生成函数、评估入口、`outputs/copy/predictions.jsonl`、`evaluation.json`，以及至少 10 个样例的分析。

**验收**：预先固定一个至少 200 条、与训练集不重复的同长度分布测试集，以整句完全匹配率 ≥95% 作为本项目目标；未达到时记录实际结果并排查，不直接宣告完成。额外测试更长序列，单独报告，不要求其达到同样指标。

完成此步，即交付了本轮核心学习项目：一个自己实现、能训练和生成、具备独立评估结果的小型 Transformer。

## 步骤 11（进阶）：接入小规模翻译

**文件**：`data/translation.py`、`configs/translation.json`、`docs/11_translation.md`；复用并扩展 `train.py`、`evaluate.py`。

**具体实现**：

1. 根据算力选择规模可控的公开双语数据，记录来源、许可、语言方向、规模与划分方式。
2. 实现文本清洗、分词、词表或 tokenizer 保存；仅用训练集拟合 tokenizer。
3. 将文本转成步骤 02 的统一批次格式，复用模型与训练流程。
4. 保存 tokenizer 和配置，确保恢复模型后得到一致的 token ID。
5. 报告翻译样例和 BLEU 等指标，并记录评价工具版本、分词设置或评价签名。

**交付物**：翻译数据流程、配置、tokenizer 文件、训练日志、checkpoint 和预测样例，放入 `outputs/translation/`；文档说明实验规模及限制。

**验收**：从原始文本到翻译输出全流程可复跑，测试数据没有用于训练；不将缩小规模或不同设置的分数直接等同于原论文结果。

## 步骤 12（进阶）：做一个消融实验并总结

**文件**：`docs/12_experiments_and_summary.md`、`outputs/ablations/`；更新 `README.md`。

**具体实现**：

1. 选择一个问题，例如去掉位置编码会怎样，或固定 D 后改变头数会怎样。
2. 固定数据划分、训练预算和评估方式，每次只改变一个因素。
3. 保存基线与变体的配置和日志，对比生成准确率、loss 和失败样例。
4. 单次实验只作为初步观察；需要更强结论时再重复多个随机种子。
5. 汇总与论文的差异：数据、规模、归一化、优化器配置、学习率调度、label smoothing、权重共享与解码方式。
6. 用自己的话说明一次前向传播、一次参数更新和一次逐词生成的全过程。

**交付物**：一个可复查的实验对比、一份整体总结、包含环境/训练/评估命令的完整 README。

**验收**：每条实验结论有实际结果支撑，能区分代码正确性、玩具任务表现和论文结果复现这三个层次。

## 交付检查清单

- [ ] 00：环境、设备与运行说明。
- [ ] 01：最小 PyTorch 训练循环。
- [ ] 02：复制任务数据、特殊符号、目标移位与 mask。
- [ ] 03：缩放点积注意力及数值检查。
- [ ] 04：多头注意力与交叉注意力形状验证。
- [ ] 05：词嵌入与位置编码。
- [ ] 06：Encoder 与源 padding 验证。
- [ ] 07：Decoder 与因果性验证。
- [ ] 08：完整 Transformer 与保存加载验证。
- [ ] 09：小批过拟合、训练日志与 checkpoint。
- [ ] 10：独立生成评估、预测样例与结果分析。
- [ ] 11（进阶）：小规模翻译实验。
- [ ] 12（进阶）：消融实验与整体总结。

当前下一步：从步骤 00 开始，确认主开发设备与现有环境。
