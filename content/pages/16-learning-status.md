这一页刻意保留"未完成"状态——诚实标注哪些内容已经在真实笔记/真实代码里验证过，哪些还没有深入，不用推测或幻觉填补空白。

## 已经吃透的部分（本站前 14 页覆盖的范围）

- 全局数据/权重/词表三条管线，以及它们如何在 `run.c` 汇合（`main()` 的四行入口）。
- Tokenizer：Python 端 SentencePiece 包装 vs C 端从零手写的 BPE 合并循环（含真实 `encode()` 全部代码），二进制格式的 byte-for-byte 对齐关系。
- 训练数据管道 `tinystories.py`：分片、`ProcessPoolExecutor` 并行、`PretokDataset` 的定长切块与打乱策略。
- `model.py` 的 `nn.Module` 内部机制：`ModelArgs` dataclass、`__setattr__`/`__getattr__`/`__call__` 三个独立钩子、`register_buffer`、weight tying 的真实赋值语义。
- 四个核心架构差异：RMSNorm、RoPE、SwiGLU、GQA/大矩阵 reshape，逐项与 nanoGPT 教学版对照，均附真实 Python + C 双语言代码。
- RoPE 的完整数学推导（旋转矩阵复合性质的代数证明）、手算数值例子、真实 `precompute_freqs_cis`/`apply_rotary_emb`/C 端现算三份代码对照。
- C 语言里"张量"的本质：flat 数组 + stride 公式，真实 `memory_map_weights()` 指针算术，`legacy_export()` 真实代码证明 layer 轴是序列化人为产物。
- `RunState` vs `TransformerWeights`：两份真实 struct 定义逐字段对照，KV cache 作为例外。
- `run.c` 的工程实现：真实 `read_checkpoint`/`mmap` 代码，`mmap` 的真实优势场景，真实 `generate()` 循环证明无 batched prefill。
- 训练工程细节：`pin_memory`/`non_blocking` 背后真实的 DMA 硬件约束、真实 `configurator.py` 全部代码。
- **（新增）`run.c` 采样策略**：真实 `sample()`/`sample_argmax()`/`sample_mult()`/`sample_topp()`/xorshift 随机数全部代码，以及与 `model.py` `generate()` 用 top-k（而非 top-p）这一真实差异。
- **（新增）学习率调度与损失评估**：真实 `get_lr()` 余弦退火公式逐行拆解，真实 `estimate_loss()` 的 `@torch.no_grad()`/`model.eval()`/多批平均设计。
- **（新增）优化器配置与 MFU**：真实 `configure_optimizers()` 按参数维度分组 decay 的判定代码、fused AdamW 的运行时反射检测，真实 `estimate_mfu()` 及其 PaLM 论文 FLOPs 公式来源。
- **（新增）PyTorch autograd 机制**：计算图/`grad_fn`/链式法则的标准机制（明确标注这是 PyTorch 框架本身的能力，非 `llama2.c` 自己的代码），混合精度 `GradScaler` 的缩放/反缩放原理，均给出可自行运行验证的最小例子。

## 尚未深入的部分（诚实清单，不在本站展开）

<div class="aside todo">
<ul>
<li><b>分布式训练（DDP）内部机制：</b><code>train.py</code> 里 <code>DistributedDataParallel</code>、<code>init_process_group</code>、<code>model.require_backward_grad_sync</code> 这套多卡梯度同步的具体实现原理，目前只知道调用方式，没有深入 PyTorch DDP 内部的 all-reduce 机制。</li>
<li><b><code>torch.compile</code> 内部机制：</b><code>train.py</code> 里 <code>model = torch.compile(model)</code> 一行背后 TorchDynamo/TorchInductor 做了什么图捕获/算子融合优化，尚未深入。</li>
<li><b>量化版本 <code>runq.c</code>：</b>仓库里与 <code>run.c</code> 并列的 int8 量化推理实现（<code>export.py</code> 里的 <code>quantize_q80</code> 已经顺带读到，但 <code>runq.c</code> 本身的量化矩阵乘法、反量化时机等尚未通读。</li>
<li><b>真实大模型 checkpoint 验证：</b>README 提到的"可以直接加载 Meta 官方 7B/13B/70B checkpoint"这条能力，尚未亲自下载真实权重跑一次验证（目前的理解完全基于 <code>ModelArgs</code> 默认值与 <code>export.py --meta-llama</code> 参数的代码证据，逻辑上成立，但没有亲手跑通）。</li>
<li><b>C++ 重写 run.c 的具体设计：</b>只有一条初步想法（用非拥有型 span/view 类封装偏移公式，避免像 <code>run.c</code> 那样在多处手写 <code>l*dim*dim</code>），还没有实际动手。</li>
<li><b>SSD 场景下 mmap 性能验证：</b>「run.c 工程实现」页里 HDD 场景的寻道劣势是从代码访问模式推理出的结论，SSD 上 mmap vs read+malloc 的实测差异是否如预期更小，还没有亲自实测过。</li>
<li><b><code>chat()</code> 对话模式与 <code>test_all.py</code> 验证体系：</b><code>run.c</code> 里 <code>chat()</code> 函数（Llama 2 Chat 模板拼接）以及仓库自带的测试套件，尚未逐行核对。</li>
</ul>
</div>

## 为什么保留这一页

这份笔记的价值恰恰在于"如实反映学习进度"——一份声称"全面讲解 llama2.c"却在关键工程细节上语焉不详或过度推测的笔记，比一份坦诚说"这几块还没学"的笔记更容易误导人。休假回来后，这份清单就是下一步学习的直接起点。
