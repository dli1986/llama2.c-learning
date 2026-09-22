这一页开始，笔记从"已经学过的部分"扩展到 STUDY_NOTES.md 尚未深入、但同样是真实工程实践的部分——采样策略、学习率调度、优化器配置、PyTorch autograd。内容依然完全来自 `llama2.c` 仓库的真实代码，不做脱离代码的推测。

## `run.c` 的 `Sampler`：三种采样策略，一层层收窄

<div data-diagram="sampling-strategy-tree" data-caption="temperature=0→贪心；否则 softmax 后走 top-p 或全概率采样"></div>

```c title="run.c -- sample(): 三选一的采样入口" hl=4,9,13,16
int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}
```

`temperature == 0`（第 4 行）时完全跳过随机性，直接 `sample_argmax`——**确定性输出**，同一个 prompt 每次生成结果都一样。一旦温度非零，先按温度缩放 logits 再 softmax（第 9 行的 `coin` 是唯一的随机源），再根据 `topp` 是否在 `(0,1)` 区间决定用全概率采样还是截断到 nucleus。

## `sample_argmax` / `sample_mult`：贪心与多项式采样

```c title="run.c -- sample_argmax()" hl=2-7
int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) { max_i = i; max_p = probabilities[i]; }
    }
    return max_i;
}
```

```c title="run.c -- sample_mult(): 按概率分布直接采样" hl=4-6
int sample_mult(float* probabilities, int n, float coin) {
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) { return i; }
    }
    return n - 1; // in case of rounding errors
}
```

`sample_mult` 是标准的**逆变换采样（inverse transform sampling）**：把 `[0,1)` 均匀随机数 `coin` 对着累积分布函数（CDF）"扎一刀"，扎中哪个区间就返回哪个 index——概率越大的 token，累积区间越宽，被扎中的概率也越大。这是离散分布采样的通用数学技巧，不是 llama2.c 专属发明。

## `sample_topp`：nucleus（top-p）采样

```c title="run.c -- sample_topp(): 只在累积概率超过 topp 的最小集合里采样" hl=8-14,17-24
int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) { last_idx = i; break; }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) { return probindex[i].index; }
    }
    return probindex[last_idx].index;
}
```

三步走：① 用 `cutoff` 粗筛掉不可能入选的低概率 token（避免对全部 32000 个 token 排序）；② 对剩下候选按概率降序排序，从高到低累加，一旦累积概率超过 `topp` 就截断在 `last_idx`；③ 只在截断后的候选集合里、按**它们之间的相对概率**重新做一次逆变换采样（`r = coin * cumulative_prob`，不是原始 `[0,1)`，因为截断后概率总和已经不是 1）。这就是"nucleus 采样只从累积概率达到 p 的最小 token 集合里采样"这句话对应的真实实现。

<div class="aside">
<b>与 <code>model.py</code> 的 <code>generate()</code> 对比：</b>PyTorch 侧的朴素生成函数用的是 <b>top-k</b>（<code>torch.topk(logits, k)</code>，只保留概率最高的固定 k 个候选），<b>不是</b> top-p。两者都是"截断长尾"的思路，但截断规则不同：top-k 截断数量固定，top-p 截断的是"凑够多少概率质量"，候选数量随分布形状浮动（分布越尖锐，nucleus 越小）。<code>run.c</code> 的真实推理管线用 top-p，训练代码里顺手写的朴素 <code>generate()</code> 用 top-k——这是同一个仓库里两条并存但不同的采样实现，不是笔误。
</div>

## `random_u32`/`random_f32`：xorshift，不依赖任何库的伪随机数

```c title="run.c -- xorshift 伪随机数生成器"
unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}
```

`Sampler` 结构体自带一个 `unsigned long long rng_state`（由 `-s` 命令行参数或 `time(NULL)` 初始化），每次采样都会推进这个状态——xorshift 是一种只用位移和异或就能生成看起来随机的数列的经典算法，不依赖 `<stdlib.h>` 的 `rand()`，也是"700 行不依赖任何库"这个目标的一部分。
