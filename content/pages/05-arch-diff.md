llama2.c 与 nanoGPT 共享同一套 Transformer 骨架，但在四个关键部件上做了不同的工程选择——每一处替换都对应 Llama 2 论文的真实设计，不是随手改动。

## 1. RMSNorm vs LayerNorm

| | LayerNorm（nanoGPT） | RMSNorm（llama2.c） |
|---|---|---|
| 归一化目标 | 均值 0、方差 1 | 只保证均方根（RMS）为 1，不管均值 |
| 计算量 | 两次遍历（mean + var） | 一次遍历（mean of squares） |
| 可学习参数 | scale(γ) + shift(β) | 只有 scale(γ) |

```c title="run.c -- rmsnorm()" hl=3
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) { ss += x[j] * x[j]; }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < size; j++) { o[j] = weight[j] * (ss * x[j]); }
}
```

上面是 C 端手写实现，下面是 `model.py` 里对应的 PyTorch 定义——两边是同一个公式，只是换了一层张量/标量的表达方式：

```python title="model.py -- RMSNorm" hl=6
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
```

## 2. RoPE vs 绝对位置 embedding

- nanoGPT：`position_embedding_table = nn.Embedding(block_size, n_embd)`，与 token embedding **相加**，是加法式、绝对位置编码。
- llama2.c：**没有位置 embedding 表**，而是把 Q/K 向量两两分组看成复平面向量，按 `角度 = pos * freq` 做旋转。数学证明与手算例子见「RoPE 深度推导」一页。

| | nanoGPT 绝对位置编码 | RoPE |
|---|---|---|
| 存储 | 一张 `(seq_len,dim)` 表，有长度上限 | 无表，现算 cos/sin |
| 作用方式 | 加到 embedding 上 | 旋转 Q、K 向量 |
| 保证的不变量 | 每个绝对位置唯一向量 | attention score 只依赖相对距离 |

## 3. SwiGLU vs 经典两层 ReLU FFN

| | nanoGPT（`bigram.py` `FeedFoward`） | llama2.c（`model.py`/`run.c` `FeedForward`） |
|---|---|---|
| 结构 | `Linear(n_embd,4n_embd)→ReLU→Linear(4n_embd,n_embd)` | `w2(SiLU(w1(x)) * w3(x))`，3 个矩阵（SwiGLU） |
| 展宽倍数 | 固定 4× | ≈2.67×（`2/3*4`），再向上取整到 `multiple_of` |
| 矩阵个数 | 2 个 | 3 个（`w1`,`w2`,`w3`） |

**`hidden_dim` 纠错**：`hidden_dim` 是 FFN 内部临时展宽的**宽度**（288→768，for stories15M），**不是层数**！层数是 `n_layers`（Block 堆叠次数），两者完全独立。**为什么叫"hidden"**：残差流（`dim`=288）贯穿所有 Block，是对外暴露的接口宽度；`hidden_dim`（768）只在 FFN 内部临时存在，算完立刻被 `w2` 压回 `dim`，不会传给下一层——"hidden"= 不进入残差流、只在子模块内部可见。

```python title="model.py -- FeedForward: SwiGLU 真实实现" hl=4-6,12
class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

C 端 `forward()` 里完全对应的三步：先算 `w1(x)` 和 `w3(x)`，再用 SiLU 手写公式处理 `w1(x)`，乘上 `w3(x)`，最后 `w2` 压回 `dim`：

```c title="run.c -- forward(): SwiGLU 非线性" hl=5
matmul(s->hb, s->xb, w->w1 + l*dim*hidden_dim, dim, hidden_dim);
matmul(s->hb2, s->xb, w->w3 + l*dim*hidden_dim, dim, hidden_dim);
for (int i = 0; i < hidden_dim; i++) {
    float val = s->hb[i];
    val *= (1.0f / (1.0f + expf(-val)));  // silu(x) = x * sigmoid(x)
    val *= s->hb2[i];
    s->hb[i] = val;
}
matmul(s->xb, s->hb, w->w2 + l*dim*hidden_dim, hidden_dim, dim);
```

## 4. 多头注意力：独立对象 vs 合并大矩阵

<div data-diagram="attention-reshape" data-caption="教学版：N 个独立 Head 对象；工程版：一次大 matmul + 指针偏移 reshape"></div>

| | nanoGPT 教学版（`bigram.py`） | llama2.c 工程版（`model.py`/`run.c`） |
|---|---|---|
| 多头怎么实现 | 真的 new 出 N 个独立 `Head` 对象，各自有自己的 K/Q/V 权重 | **一个** `wq=nn.Linear(dim, n_heads*head_dim)` 大矩阵 |
| "头"是什么 | 独立存储、独立计算 | 对同一份输出的 `.view()+.transpose()`（reshape，不拷贝） |
| 合并多头结果 | `torch.cat([h(x) for h in heads])`，N 次小 matmul | 1 次大 matmul，`run.c` 里 `s->q + h*head_size` 纯指针 offset |

**为什么工程版不用独立对象**：一次大矩阵乘法比 N 次小矩阵乘法快得多（BLAS/cache 利用率）。真实的 C 端多头循环：

```c title="run.c -- forward(): 多头注意力，指针 offset 当 reshape" hl=4,8
int h;
#pragma omp parallel for private(h)
for (h = 0; h < p->n_heads; h++) {
    float* q = s->q + h * head_size;
    float* att = s->att + h * p->seq_len;
    for (int t = 0; t <= pos; t++) {
        float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) { score += q[i] * k[i]; }
        score /= sqrtf(head_size);
        att[t] = score;
    }
}
```

`s->q + h * head_size`（第 4 行）就是 C 语言版本的 `.view(n_heads, head_size)`——同一块 `s->q` 内存，按 `head_size` 切段当成 N 个头，零拷贝。对应的 PyTorch 侧 `Attention.forward`：

```python title="model.py -- Attention.forward(): reshape 而非独立对象" hl=3-5,8
def forward(self, x, freqs_cos, freqs_sin):
    bsz, seqlen, _ = x.shape
    xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
    xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
    xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)
    xk = repeat_kv(xk, self.n_rep)  # grouped multiquery attention: expand out keys and values
    xv = repeat_kv(xv, self.n_rep)
```

**GQA（分组查询注意力）**：`n_kv_heads < n_heads` 时，`kv_mul = n_heads/n_kv_heads`，`(h/kv_mul)*head_size`（上面 C 代码第 8 行）让多个 query 头共享同一段 kv 头——本质是"多个逻辑下标映射到同一段物理内存"，类似广播但不是标准 broadcast。PyTorch 侧用 `repeat_kv()` 真正展开拷贝（非零拷贝），C 端则直接用除法得到共享的 kv 偏移，同样的语义，两种完全不同的实现手法。

## 5. 输出头与训练工程（预告，详见后续页面）

| | nanoGPT | llama2.c |
|---|---|---|
| 输出头 | 独立 `lm_head` | 与 embedding 权重共享（weight tying） |
| 优化器 | 全部参数一视同仁 | 按维度分组 decay（2D+ 权重 decay，1D 不 decay） |
| 学习率 | 固定 | warmup + 余弦衰减 |
| 训练粒度 | 每步都更新 | 梯度累积（`gradient_accumulation_steps`） |
| 混合精度/编译/DDP | 无 | `bfloat16` + `torch.compile` + `DDP` |
| 推理生成 | 无 KV cache，朴素重算 | `run.c` 用 KV cache，增量解码 |

完整的 12 维度对照表见「nanoGPT vs llama2.c 全流程对照」一页。
