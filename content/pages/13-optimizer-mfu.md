## `configure_optimizers`：按参数维度分组做 weight decay

```python title="model.py -- Transformer.configure_optimizers()" hl=6-7,9-12,17
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    param_dict = {pn: p for pn, p in self.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
    return optimizer
```

**分组规则只看张量维度**（第 6-7 行），不看参数属于哪个具体模块：

- `p.dim() >= 2`（矩阵/embedding 表，形状至少 2 维）→ 参与 weight decay。`nn.Linear.weight`、`nn.Embedding.weight` 都是 2 维，全部在这一组。
- `p.dim() < 2`（标量或 1 维向量）→ **不** decay。`RMSNorm.weight`（1 维）都在这一组——llama2.c 没有 bias（所有 `nn.Linear(..., bias=False)`），否则 bias 也会落在这一组。

这是被广泛采用的经验规则（不是 llama2.c 独创）：weight decay 的本质是对参数的 L2 范数做惩罚，鼓励参数不要过大；对矩阵类参数（真正做变换的"旋钮"）做这个惩罚是合理的正则化，但对 LayerNorm/RMSNorm 的 scale 参数或 bias（本身就该自由取值以匹配数据分布）做同样惩罚容易适得其反，所以约定俗成地把 1 维参数排除在外。

**`fused=True`**（第 17 行的前置判断）：PyTorch 的 `AdamW` 有一个"fused"实现（把每个参数张量的更新公式在 CUDA 层面合并成更少的 kernel launch），只在 GPU（`device_type == 'cuda'`）且当前 PyTorch 版本支持时才启用——`inspect.signature(...).parameters` 是运行时反射检查这个关键字参数存不存在，用来兼容旧版本 PyTorch（没有 `fused` 参数时会直接报 `TypeError`，所以必须先检查）。

## `estimate_mfu`：把训练速度换算成"占用了 GPU 理论算力的百分之几"

```python title="model.py -- Transformer.estimate_mfu()" hl=6-9
def estimate_mfu(self, fwdbwd_per_iter, dt):
    """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
    N = sum(p.numel() for p in self.parameters())
    cfg = self.params
    L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0/dt) # per second
    flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
    mfu = flops_achieved / flops_promised
    return mfu
```

`flops_per_token = 6*N + 12*L*H*Q*T`（第 6 行）这个公式**不是 llama2.c 自己推导的**，注释直接标注了出处——PaLM 论文附录 B（`https://arxiv.org/abs/2204.02311`）给出的 Transformer 前向+反向总 FLOPs 近似公式：`6N` 项对应"每个参数在一次前向+反向里大约参与 6 次浮点运算"（矩阵乘法的标准估算），`12*L*H*Q*T` 项是 attention 机制本身（跟序列长度 `T` 平方相关的部分，这里已经按每 token 摊薄）的修正项。`N`（参数总数）、`L/H/Q/T`（层数/头数/每头维度/序列长度）全部从 `self.parameters()`/`self.params`（也就是真实的 `ModelArgs`）现取，不是硬编码的估计值。

**换算链路**：每次前向+反向的 FLOPs（`flops_per_fwdbwd`）× 每次迭代做几次（`fwdbwd_per_iter`，对应梯度累积步数）÷ 这次迭代实际花的时间 `dt` = 每秒实际达到的 FLOPs（`flops_achieved`）；除以 A100 显卡 bfloat16 理论峰值算力 `312e12`，就是"实际利用率"（MFU，Model FLOPs Utilization）。这个指标是训练大模型时的标准工程诊断量——`312e12` 这个数字硬编码成 A100 的峰值，换其他型号 GPU 这个函数就需要改常量，本身是一个已知的、有意为之的简化（只针对作者训练时用的硬件）。

<div class="aside">
<b>训练循环里怎么用：</b><code>train.py</code> 每 <code>log_interval</code> 步调用一次 <code>raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)</code>，再用 <code>running_mfu = 0.9 * running_mfu + 0.1 * mfu</code> 做指数滑动平均——单次测量噪声大，滑动平均能看出稳定趋势，这跟 <code>estimate_loss()</code> 用多个 batch 平均是同一种"用统计平滑单点噪声"的工程思路。
</div>
