这一页刻意保留"未完成"状态——诚实标注哪些内容已经在真实笔记里验证过，哪些还没有深入，不用推测或幻觉填补空白。

## 已经吃透的部分（本站前 11 页覆盖的范围）

- 全局数据/权重/词表三条管线，以及它们如何在 `run.c` 汇合。
- Tokenizer：Python 端 SentencePiece 包装 vs C 端从零手写的 BPE 合并循环，二进制格式的 byte-for-byte 对齐关系。
- 训练数据管道 `tinystories.py`：分片、`ProcessPoolExecutor` 并行、`PretokDataset` 的定长切块与打乱策略。
- `model.py` 的 `nn.Module` 内部机制：`ModelArgs` dataclass、`__setattr__`/`__getattr__`/`__call__` 三个独立钩子如何接力完成参数注册与调用转发、`register_buffer` 的第三种登记方式。
- 四个核心架构差异：RMSNorm、RoPE、SwiGLU、GQA/大矩阵 reshape，逐项与 nanoGPT 教学版对照。
- RoPE 的完整数学推导（旋转矩阵复合性质的代数证明）与手算数值例子。
- C 语言里"张量"的本质：flat 数组 + stride 公式，以及 `TransformerWeights` 的 layer 轴为什么是序列化的人为产物而非 PyTorch 的真实形状。
- `RunState` vs `TransformerWeights`：激活值（1 份，反复覆写）vs 权重（n_layers 份常驻）的生命周期差异，KV cache 作为例外。
- `run.c` 的工程实现：`mmap` 的真实优势场景（并非"任何介质下都更快"）、无 batched prefill 的取舍。
- 训练工程细节：`pin_memory`/`non_blocking` 背后真实的 DMA 硬件约束、`configurator.py` 明知丑陋仍主动选择的 hack。

## 尚未深入的部分（诚实清单，不在本站展开）

<div class="aside todo">
<ul>
<li><b>run.c 解码/采样细节：</b>temperature、top-p (nucleus) 采样、repetition penalty 等 <code>sample()</code> 内部的具体策略，目前只知道"采样后生成文本"这一层，没有逐行读过采样函数实现。</li>
<li><b>train.py 训练循环剩余部分：</b><code>estimate_loss()</code>、<code>get_lr()</code> 余弦学习率调度的具体公式、<code>configure_optimizers()</code> 里 weight decay 按维度分组的判定细节、<code>estimate_mfu()</code>（模型算力利用率估算）尚未逐行对照代码验证。</li>
<li><b>PyTorch autograd 机制本身：</b><code>loss.backward()</code> 如何根据前向计算图自动求导——<code>model.py</code> 里没有手写的 backward 代码，这是一条独立的学习线，跟读 <code>forward()</code> 代码不是一回事，尚未开始。</li>
<li><b>C++ 重写 run.c 的具体设计：</b>只有一条初步想法（用非拥有型 span/view 类封装偏移公式，避免像 <code>run.c</code> 那样在多处手写 <code>l*dim*dim</code>），还没有实际动手。</li>
<li><b>SSD 场景下 mmap 性能验证：</b>「run.c 工程实现」页里 HDD 场景的寻道劣势是从代码访问模式推理出的结论，SSD 上 mmap vs read+malloc 的实测差异是否如预期更小，还没有亲自验证过。</li>
</ul>
</div>

## 为什么保留这一页

这份笔记的价值恰恰在于"如实反映学习进度"——一份声称"全面讲解 llama2.c"却在关键工程细节上语焉不详或过度推测的笔记，比一份坦诚说"这几块还没学"的笔记更容易误导人。休假回来后，这份清单就是下一步学习的直接起点。
