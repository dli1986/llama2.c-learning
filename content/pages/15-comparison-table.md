从数据源到推理生成，逐项对比 nanoGPT（`gpt-dev.ipynb`）与 llama2.c（`train.py`/`model.py`/`run.c`）在每个阶段的真实选择。

<div data-diagram="nanogpt-vs-llama2c-pipeline" data-caption="同一套 Transformer 骨架，llama2.c 每一处替换都对应真实的 Llama 2 论文设计"></div>

| # | 阶段 | `gpt-dev.ipynb`（nanoGPT） | `train.py`/`model.py`（llama2.c） |
|---|---|---|---|
| 1 | 数据源 | Tiny Shakespeare，字符级 | TinyStories，BPE |
| 2 | Tokenizer | 手写 `stoi`/`itos` | `sentencepiece` 训练的 BPE |
| 3 | 位置编码 | 绝对位置 embedding 表 | RoPE |
| 4 | 单头 attention | 独立 `Head` 对象 | 合并大矩阵 + reshape |
| 5 | 归一化 | LayerNorm | RMSNorm |
| 6 | FFN | 2 层 ReLU，4× 展宽 | SwiGLU 3 矩阵，≈2.67× 展宽 |
| 7 | 输出头 | 独立 `lm_head` | 与 embedding 权重共享（weight tying） |
| 8 | 优化器 | 全部参数一视同仁 | 按维度分组 decay（2D+ 权重 decay，1D 不 decay） |
| 9 | 学习率 | 固定 | warmup + 余弦衰减 |
| 10 | 训练粒度 | 每步都更新 | 梯度累积（`gradient_accumulation_steps`） |
| 11 | 混合精度/编译/DDP | 无 | `bfloat16` + `torch.compile` + `DDP` |
| 12 | 推理生成 | 无 KV cache，朴素重算 | `run.c` 用 KV cache，增量解码 |

## 怎么读这张表

这不是"llama2.c 处处更优"的排行榜——nanoGPT 的每一项"简化版"选择都是**故意的教学取舍**：独立 `Head` 对象让每个头的计算完全可见、可单独调试；LayerNorm/绝对位置 embedding 是 Transformer 最早、最直观的版本；固定学习率、无梯度累积让训练循环本身足够短，便于逐行讲解。

llama2.c 的每一项对应替换，都是**为了让同一套代码既能训练 TinyStories 玩具模型、又能真实加载 Meta 官方发布的 7B/13B/70B checkpoint**（见 `model.py` 里 `ModelArgs` 的默认值就是 Llama 7B 的真实架构参数）——这是工程约束逼出来的选择，不是"更先进所以选它"。两条学习路径分别展示了"从零讲透一个概念"和"贴着真实生产代码走一遍"两种不同但互补的学习方式。

<div class="aside">
<b>本站与 nanoGPT 站的关系：</b>两个站点各自独立成篇，但共享同一位作者的学习脉络——建议先看 <a href="../nanogpt-learning/index.html">nanoGPT 学习笔记</a> 打好 Transformer 基础概念，再回来看这张表，会更容易体会每一处差异背后的取舍。
</div>
