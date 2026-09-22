## `get_lr(it)`：warmup + 余弦衰减

<div data-diagram="lr-schedule-curve" data-caption="前 warmup_iters 步线性爬升，之后余弦衰减到 min_lr，超出 lr_decay_iters 后保持 min_lr"></div>

```python title="train.py -- get_lr(): 学习率调度器" hl=3-4,10
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
```

三段函数，逐段对应：

1. **`it < warmup_iters`**（第 3-4 行）：学习率从 0 线性爬升到 `learning_rate`。原因是训练刚开始时模型权重是随机初始化的，梯度方向很不稳定，直接用满额学习率容易一开始就把参数冲到很差的区域；线性爬升给优化器一个"热身"过程。
2. **`it > lr_decay_iters`**：学习率钉死在 `min_lr`（`train.py` 里设成 `learning_rate / 10`，注释标注"per Chinchilla"，即参照 Chinchilla 论文的经验设置）。
3. **中间段**（第 10 行）：`coeff` 是标准的余弦退火公式，`decay_ratio` 从 0 到 1 时，`cos(π·decay_ratio)` 从 1 降到 -1，`coeff` 从 1 平滑降到 0——所以学习率从 `learning_rate` 平滑降到 `min_lr`，且降速呈"两头慢、中间快"的 S 形（余弦曲线的导数在两端接近 0）。

这个函数每一步训练循环都会被调用一次（`lr = get_lr(iter_num) if decay_lr else learning_rate`），返回值直接赋给 `optimizer.param_groups` 里每一组的 `lr` 字段——AdamW 优化器本身不知道"调度"这回事，调度完全是外部每步手动改 `lr` 实现的。

## `estimate_loss()`：更准的 loss 估计，代价是多算几十个 batch

单个 batch 的 loss 波动很大（batch 越小波动越大），直接打印当前 batch 的 loss 不能代表模型真实水平。`train.py` 用一个独立函数专门做"更准的估计"：

```python title="train.py -- estimate_loss(): 多批平均，train/val 都测" hl=6,8-9
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        batch_iter = iter_batches(split=split)
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, Y = next(batch_iter)
            with ctx:
                logits = model(X, Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

三个关键点：

- **`@torch.no_grad()`**：这个函数只是"测量"，不需要反向传播，禁用梯度追踪能省下大量显存和计算（不用为每个中间张量保留计算图）。
- **`model.eval()` / `model.train()`**（第 4 行、第 12 行反注释掉的位置）：切换到评估模式再切回训练模式——`dropout` 在 eval 模式下会被关闭（不再随机丢弃神经元），这样测出来的 loss 才是模型"全力以赴"时的真实表现，不是带随机噪声的训练态表现。
- **`losses[k] = loss.item()`**（第 9 行的下一行）：连续测 `eval_iters`（默认 100）个 batch 取平均，用"多测几次"的统计平均换取比单个 batch 更稳定的数值——这是纯粹的工程权衡：`eval_iters` 越大，估计越准，但花的时间也越多（每次评估都要暂停训练跑 100 个 batch）。

<div class="aside">
<b>为什么 train 和 val 都要测：</b>只看 train loss 没法判断模型是不是在"死记硬背"训练数据（过拟合）；`losses["val"] < best_val_loss` 这个条件（训练循环里）决定要不要保存 checkpoint——用的是 val loss 不是 train loss，这是避免保存一个过拟合模型的标准做法。
</div>
