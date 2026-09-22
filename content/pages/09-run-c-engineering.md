## `read_checkpoint`/`mmap`：一种"裸序列化"

- **不是**带 schema 的序列化（JSON/Protobuf 那种），而是 `export.py` 按固定顺序把 header + 权重原始字节写入文件，`run.c` 按同样顺序 reinterpret，零解析。
- **`mmap` 的关键特点是"零拷贝 + 按需换页（demand paging）"**，不是"读的更快"：
  - `mmap()` 调用本身几乎不做 I/O，只在虚拟地址空间登记映射。
  - 真正的磁盘读取发生在**第一次访问**某页时（缺页中断触发）。
  - 内存页**就是**页缓存本身，不会在堆里再拷贝一份。

<div data-diagram="mmap-vs-read" data-caption="mmap 立即返回、按需换页；read+malloc 必须先阻塞读完整个文件"></div>

**与 `fread+malloc` 对比**（针对大模型，如 26GB checkpoint）：

| 维度 | `read`+`malloc` | `mmap` |
|---|---|---|
| 启动延迟 | 必须读完整个文件才能开始 | 立即返回，I/O 分摊到后续访问 |
| 内存占用 | 页缓存 + 堆内存，双倍 | 只有一份 |
| 内存压力下行为 | dirty 匿名页，驱逐需写 swap | 干净只读页，可直接丢弃重新读 |
| 多进程共享 | 各自一份 | 共享同一份物理页 |

## HDD 场景下 mmap 反而可能更慢——权重文件的物理布局是关键

`memory_map_weights` 的布局是"**按权重类型跨所有层**"整体分块（`[rms_att×6层][wq×6层][wk×6层]...`），但 `forward()` 的循环是**按层**遍历，每层都要在文件里"横跳"访问 9 个分散区域，访问完又跳回文件前部取下一层——这是拉锯式访问，不是顺序读。加上 `matmul` 会摸到**整个**权重矩阵（即使只算一个 token），所以第一次 `forward()` 调用几乎要触碰整个权重文件，只是以"来回跳"的方式，不是顺序。

**在 HDD 上，这种跳跃访问会打断内核的顺序预读（readahead）优化，可能导致大量寻道，反而比 `read+malloc` 一次性顺序读整个文件更慢**——这是"mmap 更快"这个说法容易被过度泛化的地方，它的优势场景其实是 SSD/热页缓存/多进程共享，不是任何介质下都更快。

## `run.c` 没有 batched prefill

`generate()` 的循环**每次只调用 `forward()` 处理一个 token**，即使是"消化 prompt"阶段也是逐 token 调用、只是强制把采样结果替换成已知的下一个 prompt token：

```c
while (pos < steps) {
    float* logits = forward(transformer, token, pos);   // 一次只处理一个token
    next = (pos < num_prompt_tokens - 1) ? prompt_tokens[pos+1] : sample(sampler, logits);
    ...
}
```

这跟生产级 LLM 推理（vLLM 等）"prompt 整体 batch 过一次算出全部 KV"的 prefill 优化不同——`run.c` 为了 700 行的极简目标，牺牲了这个优化。

<div class="aside">
<b>工程取舍的共同主线：</b>这一页三个决定（裸序列化、mmap 的真实优势场景、无 batched prefill）都指向同一件事——<code>run.c</code> 的设计目标是"最少代码看懂完整推理链路"，不是"生产级最优性能"。理解这个目标，才能正确判断哪些简化是有意为之、哪些是留给读者自己扩展的空间。
</div>
