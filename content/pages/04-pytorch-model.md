## `ModelArgs`：`@dataclass` + "Llama 7B 默认值"的真实含义

```python
@dataclass
class ModelArgs:
    # default hyperparameters for the Llama 7B model
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    vocab_size: int = 32000
    max_seq_len: int = 2048
    ...
```

`dim=4096, n_layers=32, n_heads=32, vocab_size=32000, max_seq_len=2048` **就是 Meta 官方 LLaMA 7B 的真实架构参数**。README 明确写了"架构与 Meta 的 Llama 2 完全一致，可以直接加载、推理 Meta 发布的真实模型"（对应 `export.py` 的 `--meta-llama` 参数）。所以这些默认值的意义是：**`ModelArgs()` 不传参数时，直接就是一份能装下真实 7B checkpoint 的正确配置**，不是随手写的示例。

在 `train.py` 训练 TinyStories 这条路径上，这些默认值**从未真正生效**——`model_args = dict(dim=dim, n_layers=n_layers, ...)` 把 `train.py` 自己的全局变量（288/6/6，给 stories15M 用）逐字段显式传入，`ModelArgs(**model_args)` 时 7B 默认值全部被覆盖。只有在某处构造 `ModelArgs()` 却没传全部字段时，7B 默认值才会真正起作用。

`@dataclass` 让你只写"字段名: 类型 = 默认值"就自动生成构造函数，`ModelArgs(**model_args)` 能正确工作，靠的正是这个自动生成的 `__init__` 按字段名匹配关键字参数——跟 `@staticmethod`（改变的是方法调用约定，不注入 `self`）解决的是完全不同的问题。

## `nn.Embedding`：可学习参数到底存在哪

`nn.Embedding` 就是一张查找表，数据存在 `.weight` 这个普通属性里：

```python
>>> embedding = nn.Embedding(10, 3)   # 10行(vocab_size)×3列(embedding_dim)
>>> embedding.weight
Parameter containing:
tensor([[ 1.0000,  1.0000,  1.0000],
        [-0.7895, -0.7089, -0.0364], ...], requires_grad=True)
```

`embedding(input)` 做的事情就是"查表"：`input` 里每个整数是"行号"，输出就是把对应那一行的向量抄出来。这跟 `run.c` 里的写法是**同一件事，只是换了层皮**：

```c
float* content_row = w->token_embedding_table + token * dim;   // C：自己手写偏移公式做查表
memcpy(x, content_row, dim*sizeof(*x));
```

## `nn.Module` 的注册机制：三个独立的 Python magic method 接力完成

<div data-diagram="nn-module-registration" data-caption="__setattr__（写）→ __getattr__（读）→ __call__（调用），三个独立钩子"></div>

- **`__setattr__`（写钩子）**：任何类都可以重写，重写后每一次 `obj.attr = value` 都会先经过这个方法。`nn.Module` 的实现：如果赋的值是 `nn.Parameter`，存进 `self._parameters`；如果是 `nn.Module`，存进 `self._modules`；否则才走进普通的 `__dict__`。**关键纠正**：`nn.Parameter`/`nn.Module` 类型的值**根本不会进 `__dict__`**，是直接绕开，不是"先进 `__dict__` 再被记一笔"。
- **`__getattr__`（读的兜底钩子）**：只在"常规方式在 `__dict__`/类属性里都找不到"时才触发，从 `self._parameters`/`self._modules` 里找回来——两条路径（手动模拟查找 vs 触发 `__getattr__` 自动查找）拿到的是**同一个对象**，因为存的是引用，不是复制。
- **`__call__`（调用钩子）**：`self.tok_embeddings(tokens)` 里的 `(tokens)` 触发的是 `nn.Module.__call__`，内部再转发到 `.forward(tokens)`。"对象能不能被 `()` 调用"是 Python 语言本身的机制（任何类定义 `__call__` 都行），"调用后去执行一个叫 `forward` 的方法"是 `nn.Module` 自己的框架约定，两者是两个独立层次。

整个模型因此形成一棵树：`Transformer → tok_embeddings(Embedding) → weight(真正的张量)`。`model.parameters()`/`model.state_dict()` 只是**递归遍历这棵树**，自动收集所有登记过的参数，不需要手动一个个列出来。

<div class="aside warn">
<b>容易搞混的点：</b><code>h = self.tok_embeddings(tokens)</code> 里的 <code>h</code> 是普通局部变量赋值，从未经过 <code>self.xxx=</code>，<code>__setattr__</code> 的检查机制根本没被触发过——不是"检查了发现不是 Parameter 所以被排除"。"hidden"这个名字只是深度学习传统术语（隐藏层：不直接暴露给外部的中间表示），跟"是否被 <code>_parameters</code> 记录、是否被 optimizer 更新"没有因果关系，这是两个完全独立的维度。
</div>

## `register_buffer`：第三种登记方式

`model.py` 里的 RoPE 频率表：

```python
freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
self.register_buffer("freqs_cos", freqs_cos, persistent=False)
self.register_buffer("freqs_sin", freqs_sin, persistent=False)
```

存的是第三个独立字典 `_buffers`——同样不会出现在 `__dict__` 里。跟 `nn.Parameter` 的关键区别：

| | `nn.Parameter`（如 `tok_embeddings.weight`） | `register_buffer`（如 `freqs_cos`） |
|---|---|---|
| 会不会被 optimizer 更新 | 会 | **不会** |
| 会不会跟 `.to(device)` 走 | 会 | 会（所以不能用普通属性存） |
| 会不会存进 `state_dict()` | 会 | 这里 `persistent=False` → 明确设成不存，因为随时能用 `(head_dim, max_seq_len, theta)` 重新算出来，没必要占 checkpoint 体积 |
| 为什么需要显式注册 | 类型判断自动路由 | 普通 `Tensor` 光看类型无法区分"该不该是 buffer"，必须手动调用表态 |
