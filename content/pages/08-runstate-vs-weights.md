## 两者的定位对照

| | `TransformerWeights` | `RunState` |
|---|---|---|
| 内容 | 训练学到的、只读、固定的参数 | forward() 过程中算出来的中间结果 |
| 每层几份 | `n_layers` 份（拼接在一个大数组里） | **只有 1 份**，每层循环反复覆写 |
| 生命周期 | 从 checkpoint 加载后不变 | 一次 forward 调用内产生、消费、丢弃 |

<div data-diagram="runstate-vs-weights" data-caption="权重 n_layers 份常驻；激活值只 1 份，像水波一样反复覆写"></div>

**"current wave of activations"**：`x`/`xb`/`hb` 等 buffer 只分配一份，`for l in 0..n_layers` 循环每次都覆写同一块内存——像水波一样冲刷过去，不像权重那样 n_layers 份并存。

## "activation" 术语辨析

- **activation function**（激活函数）= ReLU/SiLU/softmax 这类具体非线性函数。
- **activation(s)**（激活值）= 前向传播中算出的任何中间张量，泛指"不是权重的东西"。
- `RunState` 里大多数字段（`q`/`k`/`v`/`x`/`xb`/`xb2`/`logits`/两个 cache）**都没有经过字面意义的激活函数**（纯线性变换/加法），但依然属于"activation"这个大类——判断标准是"计算出来的 vs 权重"，不是"过没过非线性函数"。真正过了字面激活函数的只有 `hb`（SiLU）和勉强算的 `att`（softmax）。

## 例外：`key_cache`/`value_cache`——只追加，不覆写

因为 attention 需要回看所有历史位置的 k/v，KV cache **既有 layer 轴也有 position 轴**，新 token 只追加、旧的不清空，跟其它 activation buffer 的"每层覆写一次"行为完全不同。

`att` 的形状 `(n_heads, seq_len)`：存的是**分数（标量）**，不是向量，所以第二维是 `seq_len`（历史位置数量）而非 `head_size`（单头向量宽度）——两者是完全不相关的两个数（stories15M 里 `head_size=48`，`seq_len=256`）。

<div class="aside evidence">
<b>为什么这个区分重要：</b>理解 <code>RunState</code> vs <code>TransformerWeights</code> 的生命周期差异，是理解「run.c 工程实现」页里 mmap 加载策略、以及 KV cache 增量解码为什么能把逐 token 推理复杂度从 O(n²) 降到近似 O(n) 的前提——它们分别对应"不变的常量"和"每次 forward 都要重新流过的状态"。
</div>
