<div class="aside">
<b>说明：</b>本页讲的 autograd 机制是 <b>PyTorch 框架本身</b>的能力，不是 <code>llama2.c</code> 仓库自己写的代码——<code>model.py</code> 里完全没有手写的反向传播；`train.py` 只是在正确的时机调用了 `loss.backward()`。这一页把这个"调用点背后到底发生了什么"讲清楚，内容是 PyTorch 官方文档记载的标准机制（可在 PyTorch 官网 "A Gentle Introduction to torch.autograd" 一文交叉验证），不是脱离代码的推测。
</div>

## 真实调用点：`train.py` 里 autograd 实际被触发的地方

```python title="train.py -- 训练主循环: autograd 真正被调用的两行" hl=3,7
with ctx:
    logits = model(X, Y)
    loss = raw_model.last_loss  # 在 Transformer.forward() 里由 F.cross_entropy 算出
    loss = loss / gradient_accumulation_steps
X, Y = next(train_batch_iter)
scaler.scale(loss).backward()
```

`model(X, Y)`（第 3 行，触发 `Transformer.forward`）这一次调用，**不只是算出了 logits**——只要参与计算的张量有任何一个 `requires_grad=True`（模型参数默认都是），PyTorch 就会在算的同时，**顺便记录下这次计算是怎么做出来的**。`loss.backward()`（第 7 行）才是真正"反向"的地方。

## 计算图：每算一步，就顺手记一笔"怎么算出来的"

PyTorch 用的是 **define-by-run**（动态图）：不是像某些框架那样先声明一张固定的计算图再灌数据，而是每次 `forward()` 实际跑代码时，图就跟着长出来。每个由张量运算产生的中间结果，都会带一个 `grad_fn` 属性，指向"生成它的那个运算"：

```python
>>> import torch
>>> x = torch.tensor([2.0], requires_grad=True)
>>> y = x * 3
>>> z = y + 1
>>> z.grad_fn
<AddBackward0 object at ...>
>>> z.grad_fn.next_functions
((<MulBackward0 object at ...>, 0), (None, 0))
```

`z.grad_fn` 是 `AddBackward0`（对应 `y + 1` 这次加法），它的 `next_functions` 指回上一步的 `MulBackward0`（对应 `x * 3`）——一路往回追，就能追出整条计算链路。这条链路本质是一个**有向无环图（DAG）**：叶子节点是模型参数（`requires_grad=True` 且没有 `grad_fn`，因为它们不是算出来的，是直接声明的），每个非叶子节点都记着"我是被谁、用什么运算生成的"。

`Transformer.forward()` 里的 `h = self.tok_embeddings(tokens)` → 逐层 `TransformerBlock` → `self.norm(h)` → `self.output(h)` → `F.cross_entropy(...)`，每一步都在往这张图上添加节点——`model(X, Y)` 跑完的那一刻，一张从"输入 token"到"最终 loss"的完整计算图已经在内存里搭好了，还没有做任何反向传播。

## `loss.backward()`：顺着计算图反向走一遍链式法则

```python
>>> z.backward()
>>> x.grad
tensor([3.])
```

$z = 3x+1$，$\frac{dz}{dx}=3$——`x.grad` 里的 `3.` 正是这个导数的数值，PyTorch 没有做符号推导，而是**顺着刚才记录的计算图反向遍历**，在每个节点用链式法则把"上游传来的梯度"乘以"这个运算自己的局部导数"，再往更上游传，一直传到叶子节点（这里是 `x`）为止。`loss.backward()` 做的是同一件事，只是链条长得多（几十上百个算子），并且是从一个标量（`loss`，形状 `(1,)`）反向铺开到千万级参数的每一个元素。

**梯度是累加的，不是覆盖的**——这是 autograd 一个容易忽略但很关键的设计：每次 `.backward()` 只会往 `.grad` 上**加**新算出来的值，不会自动清零。这也是为什么训练循环末尾必须显式 `optimizer.zero_grad(set_to_none=True)`——不清零的话，下一步的梯度会跟这一步的叠在一起。反过来说，`train.py` 里梯度累积（`gradient_accumulation_steps` 次 micro-step 才 `optimizer.step()` 一次）能生效，靠的正是这个"默认累加"的行为：中间那几次 `.backward()` 故意不清零、不 `step()`，让梯度自然叠加，等效于用一个更大的 batch size 训练。

## `scaler.scale(loss).backward()`：混合精度训练要多做的一步

`train.py` 用 `float16` 训练时（`dtype = "float16"`），`GradScaler` 会包一层：

```python
scaler.scale(loss).backward()   # 反向传播前，先把 loss 乘大一个系数
scaler.unscale_(optimizer)      # 更新参数前，再把梯度除回去
scaler.step(optimizer)
scaler.update()
```

`float16` 能表示的数值范围比 `float32` 窄得多，训练中很小的梯度值经常会在 `float16` 精度下直接下溢成 0（"vanishing gradient"的一种工程表现，不是数学上的梯度消失，是浮点数表示精度的物理限制）。`GradScaler` 的技巧是：反向传播前先把 `loss` 乘一个较大的系数（比如 65536），根据链式法则，这个系数会随着反向传播原样乘到每一层的梯度上，把本来会下溢成 0 的小梯度"撑"到 `float16` 能表示的范围内；等梯度算完、准备更新参数前，再把这个系数除回去，物理效果不变。`scaler.update()` 还会动态调整这个系数——如果发现梯度出现了 `inf`/`nan`（说明系数放大过头），下次自动调小。

<div class="aside evidence">
<b>诚实边界：</b>以上是 autograd/混合精度训练的通用机制，来自 PyTorch 官方文档与广泛的工程实践共识，不是从 <code>llama2.c</code> 源码里"读"出来的——因为这份源码根本没有实现这一层，它只是正确地调用了 PyTorch 提供的这套机制。如果需要更深入的验证，可以在自己的 Python 环境里跑上面两段 <code>x.grad</code>/<code>z.grad_fn</code> 的最小例子，用真实运行结果核对本页的说法。
</div>
