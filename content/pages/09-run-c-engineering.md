## `read_checkpoint`/`mmap`：一种"裸序列化"

- **不是**带 schema 的序列化（JSON/Protobuf 那种），而是 `export.py` 按固定顺序把 header + 权重原始字节写入文件，`run.c` 按同样顺序 reinterpret，零解析。
- **`mmap` 的关键特点是"零拷贝 + 按需换页（demand paging）"**，不是"读的更快"：
  - `mmap()` 调用本身几乎不做 I/O，只在虚拟地址空间登记映射。
  - 真正的磁盘读取发生在**第一次访问**某页时（缺页中断触发）。
  - 内存页**就是**页缓存本身，不会在堆里再拷贝一份。

```c title="run.c -- read_checkpoint(): mmap 一整个 checkpoint 文件" hl=13
void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { exit(EXIT_FAILURE); }
    fread(config, sizeof(Config), 1, file);
    int shared_weights = config->vocab_size > 0 ? 1 : 0;
    config->vocab_size = abs(config->vocab_size);
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    *fd = open(checkpoint, O_RDONLY);
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr, shared_weights);
}
```

`fopen`+`fread` 只读了固定大小的 `Config` 头部；真正的权重数据完全靠第 13 行的 `mmap()` 登记映射，一个字节都还没被复制到堆内存——`memory_map_weights`（见「C 语言里的"张量"」页）拿到的 `weights_ptr` 只是这块映射内的一个偏移起点。

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

```c title="run.c -- generate(): 逐 token 循环，无 batched prefill" hl=6
int token = prompt_tokens[0];
int pos = 0;
while (pos < steps) {
    float* logits = forward(transformer, token, pos);

    if (pos < num_prompt_tokens - 1) {
        next = prompt_tokens[pos + 1];   // 仍在消化 prompt：强制用已知的下一个 token
    } else {
        next = sample(sampler, logits);   // 已进入自由生成：真正采样
    }
    pos++;
    if (next == 1) { break; }   // BOS 终止序列

    token = next;
}
```

第 6 行是关键：`forward()` 每次调用只喂 1 个 token（配合 `pos` 走 KV cache 增量），无论是在"重放 prompt"还是"自由生成"阶段，调用形状完全一样——区别只在于 `next` 的来源是数组里的已知答案还是 `sample()` 的真实采样。这跟生产级 LLM 推理（vLLM 等）"prompt 整体 batch 过一次算出全部 KV"的 prefill 优化不同——`run.c` 为了 700 行的极简目标，牺牲了这个优化。

<div class="aside">
<b>工程取舍的共同主线：</b>这一页三个决定（裸序列化、mmap 的真实优势场景、无 batched prefill）都指向同一件事——<code>run.c</code> 的设计目标是"最少代码看懂完整推理链路"，不是"生产级最优性能"。理解这个目标，才能正确判断哪些简化是有意为之、哪些是留给读者自己扩展的空间。
</div>
