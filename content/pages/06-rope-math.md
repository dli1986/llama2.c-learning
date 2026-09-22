这一页是对「模型结构差异」里 RoPE 一节的深入补充，对着 `model.py` 里 `precompute_freqs_cis`/`apply_rotary_emb`/`register_buffer` 的真实代码，把"为什么用旋转"讲到数学证明层面，并用一个可以手算的小例子把"多频率"这件事钉死。

## 数学核心：为什么"旋转"能让 attention score 只依赖相对距离

<div data-diagram="rope-rotation" data-caption="q 转 mθ、k 转 nθ，点积后夹角只剩 (n-m)θ"></div>

**几何直觉**：旋转矩阵

$$R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{pmatrix}$$

只改变一个二维向量的**朝向**，不改变它的**长度**（正交变换的定义性质）。

**核心恒等式**（旋转矩阵的复合性质）：

$$R(\alpha)^T R(\beta) = R(\beta-\alpha)$$

（先转 $\alpha$ 再转 $\beta$ 的复合效果等于直接转 $\beta-\alpha$；转置 $R(\alpha)^T$ 等价于反着转 $R(-\alpha)$。）

**推导**：位置 $m$ 的 query 先转 $m\theta$ 得到 $q'=R(m\theta)q$，位置 $n$ 的 key 转 $n\theta$ 得到 $k'=R(n\theta)k$，两者点积：

$$q'\cdot k' = q^T R(m\theta)^T R(n\theta)k = q^T R\big((n-m)\theta\big)k$$

**结果只剩 $(n-m)$，绝对位置 $m,n$ 本身从公式里消失了**——这是"attention score 只依赖相对距离"的**代数证明**，不是拍脑袋设计的规律。

**对比绝对位置编码**：$x_m=\text{tok}_m+\text{pos}_m$，点积 $q\cdot k$ 展开是"内容·内容 + 内容·位置 + 位置·内容 + 位置·位置"四项交叉相加，**没有干净的数学结构能保证"只依赖相对距离"**，模型只能在训练里慢慢近似学出类似的规律，不是架构上写死的保证。

## 数学落到代码：PyTorch 侧 vs C 侧，同一套旋转的两种实现

先建频率表（位置信息在这一步就已经烘焙进去了），再在 `apply_rotary_emb` 里对 Q/K 做旋转：

```python title="model.py -- precompute_freqs_cis(): 建频率表" hl=2
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin
```

```python title="model.py -- apply_rotary_emb(): 对 Q/K 做旋转" hl=10-11
def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos
    return torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3), \
           torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)
```

C 端没有预先建表，而是**逐 token 现算** `cos`/`sin`（推理只需要一个位置，不值得为整张表付出内存/带宽）：

```c title="run.c -- forward(): RoPE 现算 cos/sin 并旋转 q、k" hl=3-5,11-13
for (int i = 0; i < dim; i+=2) {
    int head_dim = i % head_size;
    float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    float val = pos * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    int rotn = i < kv_dim ? 2 : 1; // how many vectors? 2 = q & k, 1 = q only
    for (int v = 0; v < rotn; v++) {
        float* vec = v == 0 ? s->q : s->k;
        float v0 = vec[i];
        float v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
    }
}
```

`fcr = cos(pos*freq)`、`fci = sin(pos*freq)` 就是 PyTorch 侧 `freqs_cos[pos]`/`freqs_sin[pos]` 这一行的现算版本；`rotn = i < kv_dim ? 2 : 1` 是 GQA 的直接体现——当 `dim`（q 的宽度）比 `kv_dim`（k 的宽度）大时，超出 `kv_dim` 的部分只旋转 q，不旋转 k（因为这部分 k 根本不存在）。

## RoPE 的额外好处

| 好处 | 原因 |
|---|---|
| 数学上**保证**相对位置性质，不是训练近似出来的 | 见上面的代数推导 |
| 长度外推能力更强 | 角度=位置×频率是连续函数，遇到训练时没见过的位置也能算出合理角度；绝对位置 embedding 是固定行数的表，超出行数直接没法处理 |
| 不破坏 q/k 的模长语义 | 旋转是正交变换，不改变 $\lVert q\rVert$/$\lVert k\rVert$，只改变朝向；绝对位置是加法，会直接改变模长和方向，和内容信息搅在一起 |
| 多频率同时编码"细粒度"和"粗粒度"位置关系 | 见下面的具体算例 |

位置编码方式和 causal mask 是**两件独立的事**：绝对位置编码、RoPE，两者都要配合 causal mask（上三角 `-inf`，或 `is_causal=True`）才能保证"不看未来"——causal mask 解决"允许看谁"，位置编码方式解决"怎么让模型知道位置关系"，不要混在一起理解。

## 具体算例：`head_dim=4`（2 对频率），4 个位置，手算旋转过程

```python
>>> freqs = 1.0 / (10000 ** (torch.arange(0,4,2) / 4))
tensor([1.0000, 0.0100])          # 频率0=1.0(快)，频率1=0.01(慢)
>>> t = torch.tensor([0,1,2,3])
>>> freqs = torch.outer(t, freqs)
tensor([[0.0000, 0.0000],
        [1.0000, 0.0100],
        [2.0000, 0.0200],
        [3.0000, 0.0300]])
>>> freqs_cos = torch.cos(freqs)
tensor([[ 1.0000,  1.0000],
        [ 0.5403,  0.9999],
        [-0.4161,  0.9998],
        [-0.9900,  0.9996]])
>>> freqs_sin = torch.sin(freqs)
tensor([[0.0000, 0.0000],
        [0.8415, 0.0100],
        [0.9093, 0.0200],
        [0.1411, 0.0300]])
```

**关键澄清（容易搞混的点）**：`q=[1,0,1,0]` 是**一个 token、坐在某个位置上**的 query 向量本身的 4 个数（`head_dim=4`），**不是"4 个不同位置的 token id"**——"位置"和"q 向量内部的维度"是两条完全独立的轴：

- "位置"决定去 `freqs_cos`/`freqs_sin` 表里查**第几行**。
- `head_dim` 决定 `q` 自己被拆成**几对**、每对配**第几列**（拆分方式固定，不随位置变）。

假设这个 token 坐在位置 2，取表的第 2 行：`freqs_cos[2]=[-0.4161, 0.9998]`，`freqs_sin[2]=[0.9093, 0.0200]`。

**配对规则（容易出错的地方）**：必须是**同一个频率自己的 cos 和 sin 配一对**，不是"两个频率的 cos 配一对、sin 配另一对"：

| | 频率0（快）在位置2的 (cos,sin) | 频率1（慢）在位置2的 (cos,sin) |
|---|---|---|
| 正确配对 | (-0.4161, 0.9093) | (0.9998, 0.0200) |

`q` 拆成 2 对：第 0 对 $(q_0,q_1)=(1,0)$ 用频率 0，第 1 对 $(q_2,q_3)=(1,0)$ 用频率 1：

$$\text{第0对}:\begin{pmatrix}-0.4161&-0.9093\\0.9093&-0.4161\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix}=\begin{pmatrix}-0.4161\\0.9093\end{pmatrix}$$

$$\text{第1对}:\begin{pmatrix}0.9998&-0.0200\\0.0200&0.9998\end{pmatrix}\begin{pmatrix}1\\0\end{pmatrix}=\begin{pmatrix}0.9998\\0.0200\end{pmatrix}$$

拼回 4 维：`q` 从 `[1,0,1,0]` 变成 `[-0.4161, 0.9093, 0.9998, 0.0200]`。

**高频/低频的具体体现**：同一个位置（2），第 0 对（频率 1.0）转了 2.0 弧度（约 114.6°，朝向变化很大），第 1 对（频率 0.01）只转了 0.02 弧度（约 1.15°，几乎没变）——**高频对相邻位置的细微差异敏感，低频对只有隔很远的位置才会有明显差异**，类似二进制多个 bit 位同时表达不同数量级的信息。

## 最核心的一句话总结

**位置信息在"建频率表"（`torch.outer(t, freqs)`）这一步就已经烘焙进 `freqs_cos`/`freqs_sin` 里了**——`apply_rotary_emb` 里的旋转公式本身是通用的、不知道"位置"这个概念的纯数学运算，它只是老实地拿"查表查出来的某一行 cos/sin"去转 q/k。`register_buffer` 存的正是这张"提前算好、位置信息已经编码在内"的表，之后每次 forward 只是**查表复用**，不需要重新计算。

## `forward()` 里一个容易忽略的细节：切片不是查表

```python
h = self.tok_embeddings(tokens)          # gather：tokens 里的 id 可以是任意值、任意顺序、可重复
freqs_cos = self.freqs_cos[:seqlen]      # slice：永远是"第0行到第seqlen-1行"这一段连续区间
```

`model.py` 的 `forward()`（训练和 `generate()` 都一样）**每次都是把当前这整段序列从头到尾重新完整算一遍**（没有 KV cache），所以"这批 token 的位置"永远是"从 0 数到 seqlen-1"，直接切片 `[:seqlen]` 天然对应。对比 `run.c`：`run.c` 显式维护 `pos` 变量，因为它是逐 token 增量生成（KV cache 已经攒了 N 个，新 token 的位置就是 N，不是从 0 开始）。
