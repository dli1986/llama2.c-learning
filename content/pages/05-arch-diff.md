llama2.c 与 nanoGPT 共享同一套 Transformer 骨架，但在四个关键部件上做了不同的工程选择——每一处替换都对应 Llama 2 论文的真实设计，不是随手改动。

## 1. RMSNorm vs LayerNorm

| | LayerNorm（nanoGPT） | RMSNorm（llama2.c） |
|---|---|---|
| 归一化目标 | 均值 0、方差 1 | 只保证均方根（RMS）为 1，不管均值 |
| 计算量 | 两次遍历（mean + var） | 一次遍历（mean of squares） |
| 可学习参数 | scale(γ) + shift(β) | 只有 scale(γ) |

```c
void rmsnorm(float* o, float* x, float* weight, int size) {
    float ss = 0.0f;
    for (int j = 0; j < size; j++) ss += x[j] * x[j];   // 只算平方和，不减均值
    ss = 1.0f / sqrtf(ss/size + 1e-5f);
    for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
}
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

## 4. 多头注意力：独立对象 vs 合并大矩阵

<div data-diagram="attention-reshape" data-caption="教学版：N 个独立 Head 对象；工程版：一次大 matmul + 指针偏移 reshape"></div>

| | nanoGPT 教学版（`bigram.py`） | llama2.c 工程版（`model.py`/`run.c`） |
|---|---|---|
| 多头怎么实现 | 真的 new 出 N 个独立 `Head` 对象，各自有自己的 K/Q/V 权重 | **一个** `wq=nn.Linear(dim, n_heads*head_dim)` 大矩阵 |
| "头"是什么 | 独立存储、独立计算 | 对同一份输出的 `.view()+.transpose()`（reshape，不拷贝） |
| 合并多头结果 | `torch.cat([h(x) for h in heads])`，N 次小 matmul | 1 次大 matmul，`run.c` 里 `s->q + h*head_size` 纯指针 offset |

**为什么工程版不用独立对象**：一次大矩阵乘法比 N 次小矩阵乘法快得多（BLAS/cache 利用率）。`run.c` 的 `s->q + h * head_size` 就是 C 语言版本的 `.view(n_heads, head_dim)`。

**GQA（分组查询注意力）**：`n_kv_heads < n_heads` 时，`kv_mul = n_heads/n_kv_heads`，`(h/kv_mul)*head_size` 让多个 query 头共享同一段 kv 头——本质是"多个逻辑下标映射到同一段物理内存"，类似广播但不是标准 broadcast。

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
