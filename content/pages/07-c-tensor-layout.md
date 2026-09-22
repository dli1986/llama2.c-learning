## 核心公式：任何形状都是 flat `float*` + 步长（stride）公式

<div data-diagram="tensor-stride-layout" data-caption="offset = l·(dim·dim) + i·dim + j —— C 端手写偏移公式做查表"></div>

$$\text{offset}(i_1,\dots,i_k) = i_1\cdot(d_2\cdots d_k) + \cdots + i_k$$

`float* buffer` 本身**不携带任何形状信息**，`(vocab_size,dim)` 还是 `(layer,dim,dim)` 完全是代码里怎么解释这串偏移公式。三个实例：

| 代码 | 逻辑形状 | 偏移公式 |
|---|---|---|
| `matmul` 里 `w[i*n+j]` | `(d,n)` | $i\cdot n+j$ |
| `w->wq + l*dim*dim` | `(n_layers,dim,dim)` | $l\cdot(dim\cdot dim)+i\cdot dim+j$ |
| `key_cache + loff + t*kv_dim` | `(n_layers,seq_len,kv_dim)` | $l\cdot(seq\_len\cdot kv\_dim)+t\cdot kv\_dim+\dots$ |

真实的指针行进过程——`memory_map_weights` 不做任何拷贝，只是把一个 `float*` 指针按照权重大小依次往前挖：

```c title="run.c -- memory_map_weights(): 纯指针算术，零拷贝" hl=6-7,9-10
void memory_map_weights(TransformerWeights *w, Config* p, float* ptr, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim;
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size);
    // ... wv, wo, rms_ffn_weight, w1, w2, w3 依次同理
    w->rms_final_weight = ptr;
    ptr += p->dim;
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}
```

没有任何 `malloc`/拷贝——每一行只是把 `ptr` 指向文件里的下一段地址，再把指针向前挖过刚才那个字段的总大小（`p->vocab_size * p->dim`、`n_layers * p->dim * (...)` …）。这个函数自身就是“偏移公式只写一处”的实例——每个字段的起始地址都是上一个字段的终点，顺序必须与 `export.py` 写入的顺序一致，否则整个模型会静默错位。

## shape-only vs shape+stride：不是记法不同，是表达力不同

只存 shape、靠"标准连续假设"现算 stride，**只能表达完全连续的布局**。以下场景必须显式存 stride（与 shape 分离）才能表达：

| 场景 | 只有 shape | 显式 stride |
|---|---|---|
| 转置（transpose） | 做不到（必须拷贝） | `stride=(1,d0)`，零拷贝 |
| 隔层/跳步抽取 | 做不到 | `stride0 = 2*(d1*d2)` |
| 广播（broadcast） | 做不到 | `stride=0` |

PyTorch 的 `Tensor` 独立维护 `.shape` 和 `.stride()` 正是因为这个原因——`transpose()` 零拷贝就是把 stride 顺序倒过来，shape 变了但底层数据没动。

## `TransformerWeights` 的 shape 注释：小心"读法" vs "存法"

注释 `wq: (layer, dim, n_heads*head_size)` 写的是"输入维度→输出维度"的**人话描述**（对应 `nn.Linear(dim, n_heads*head_size)` 构造函数参数顺序），但**真实内存布局/`matmul` 按 `(d,n)` 解读的顺序是反的**（`n_heads*head_size, dim`），跟 PyTorch 真正的 `.weight.shape=(out_features,in_features)` 一致。这是文档书写习惯 vs 实际存储顺序的一个典型不一致案例，读代码时不能只看注释字面意思。

## 指针 offset：保证性能，不保证正确性

- **保证的是性能**：O(1) 算术定位，零拷贝。
- **不保证正确性**：完全依赖"写入偏移公式"和"读取偏移公式"手工保持一致，没有任何格式层面的校验。一个公式写错，不会报错，只会静默读出错位的垃圾数据。C++ 可以用封装（把偏移公式只写一处，比如构造函数里）来降低这个风险，但"零拷贝、指针算术"这个底层机制本身不会因为用了 class 就自动变得更安全。

## "layer"轴是 C 端序列化的人为产物，不是 PyTorch tensor 的真实形状

- 激活值 `h`：`(B,T,C)`，在 `for layer in self.layers` 循环里被反复覆写，没有 layer 轴。
- 权重：`self.layers` 是 `nn.ModuleList`，每层是**独立 Python 对象**（`self.layers[0].attention.wq.weight` 各自形状 `(dim,dim)`），PyTorch 从没把它们拼成一个带 layer 轴的大 tensor。
- `run.c` 注释里的 `layer` 轴，来自 `export.py` 把 `n_layers` 个独立小矩阵**首尾拼接**进一个文件的动作——是 C 端为了用一个 mmap 指针 + 偏移量寻址，人为制造出来的轴：

```python title="export.py -- legacy_export(): layer 轴是这样被“拼”出来的" hl=2,4
# attention weights
for layer in model.layers:
    serialize_fp32(out_file, layer.attention_norm.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wq.weight)
for layer in model.layers:
    serialize_fp32(out_file, layer.attention.wk.weight)
```

`model.layers` 是 `nn.ModuleList`，循环里每次 `layer.attention.wq.weight` 都是一个独立形状 `(dim,dim)` 的 Python 张量对象——**没有任何一个 PyTorch 张量真正拥有 `(n_layers,dim,dim)` 这个形状**。`for layer in model.layers: serialize_fp32(...)`（第 2/4 行）逐层写入文件的这个动作，才是 `run.c` 里 `(layer, dim, dim)` 这个形状注释的真正来源——导出时才产生的序列化人为产物，不是 PyTorch tensor 本身的形状。
- 唯一没有 `layer` 前缀的 `token_embedding_table`/`rms_final_weight`，对应 `model.py` 里**不属于** `self.layers`、只在 `Transformer` 类里出现一次的字段（`self.tok_embeddings`、`self.norm`）。

## Tensor 不是 list——底层永远是"一整块连续内存"

```python
>>> input = torch.tensor([[1,2,4,5],[4,3,2,9]])
>>> input.shape
torch.Size([2, 4])
```

| | Python `list` | `torch.Tensor` |
|---|---|---|
| 内存布局 | 一堆指针，每个指向别处一个独立对象，可混装不同类型 | **一整块连续内存**，同一种类型的原始数字 |
| 多维怎么表达 | 靠 list 嵌套 list 这个**真实存在**的结构 | 靠 flat 内存 + shape/stride 公式，跟嵌套结构无关 |
| 构造语法里的嵌套 | 就是真实结构本身 | 只是给 `torch.tensor()` 看的语法糖，造出来之后嵌套就"拉平"消失了 |
| `x[i]` 取到的是什么 | 一个**本来就独立存在**的对象 | 一个新计算出来的 **view**——共享同一块底层内存，没有拷贝 |

`torch.tensor([[1,2,4,5],[4,3,2,9]])` 这个 `(2,4)` 张量，底层就是一整块内存 `[1,2,4,5,4,3,2,9]` 加上"形状是 `(2,4)`"这份元数据——`input[0,2]` 取到 `4`，本质是按 `offset=0*4+2=2` 去这块 flat 内存取第 2 个位置的值，跟 `run.c` 里 `w[i*n+j]`、`l*dim*dim` 是**同一套底层原理**。这也是为什么 `tensor[i]` 改了会影响原数据（共享内存的 view），而 `list[i]` 改了通常不会牵连外层——两者"看起来都能用 `[i]` 取子结构"，但语义完全不同。
