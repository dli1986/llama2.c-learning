## 一句话概括

`stories15M.bin` 是 Karpathy 用仓库自带的 `train.py` 在 TinyStories 数据集上训练、再用 `export.py` 导出的 checkpoint；执行 `./run stories15M.bin` 是**一次推理**（自回归生成），不是训练。命令行打印的 `achieved tok/s` 是推理吞吐量，不是训练速度。

## 完整链路

<div data-diagram="pipeline-overview" data-caption="三条独立的数据/模型管线，最终在 run.c 里汇合"></div>

三条管线各自独立，最后在 C 端汇合：

1. **训练数据**：TinyStories 原始故事文本（`.json`）→ `tinystories.py` 的 `pretokenize()` → 50 个 `dataXX.bin`（token id 序列）→ 被 `train.py` 训练时消费。
2. **模型权重**：`train.py`（PyTorch 训练：forward → loss → backward → `optimizer.step()`）→ `export.py` → `model.bin`（权重，C 端可 `mmap` 读取的裸二进制）。
3. **词表**：`tokenizer.model`（Meta 预训练的 SentencePiece 词表）→ `tokenizer.py` 的 `export()` → `tokenizer.bin`（C 端可读的词表格式）。

最终 `run.c` 加载 `model.bin` + `tokenizer.bin`，进入 `forward(token, pos)` 的逐 token 前向推理循环，采样后生成文本。

## 为什么要看这份代码

`run.c` 全文约 700 行纯 C，没有任何深度学习框架依赖，却能加载并运行与 PyTorch 训练完全等价的 Transformer 权重。这意味着：模型结构本身可以完全脱离 PyTorch 的自动求导/张量库描述——权重不过是一串 `float`，前向传播不过是几个手写的矩阵乘法 + 归一化 + 激活函数。看懂 `run.c`，等于看懂了"Transformer 推理"这件事去掉框架糖衣之后剩下的全部本质。

<div class="aside">
<b>本站的组织方式：</b>不是按 STUDY_NOTES.md 原始的问答顺序，而是重新划分成「Python/PyTorch 侧 → C 侧内存与工程 → 与 nanoGPT 的逐项对照 → 诚实的学习进度」四个板块，每页一个主题，方便按需查阅。
</div>
