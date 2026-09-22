## 三个阶段

`tinystories.py` 把原始故事文本变成训练可用的 token id 序列，分三步：

1. **`download()`**：下载解压 TinyStories 为 50 个 `.json` shard。
2. **`train_vocab(vocab_size)`**：**可选**，只有 `vocab_source="custom"` 时才需要——用 `SentencePieceTrainer` 在语料上**训练**一套新的 BPE 子词规则，`vocab_size` 是**设定的目标**，不是数出来的。默认 `vocab_source="llama2"` 时**跳过这一步**，直接用 Meta 官方 `tokenizer.model`（32000 token）。
3. **`pretokenize(vocab_size)`**：调用 `process_shard` 把每个 shard 的故事文本编码成 token id，写成 `dataXX.bin`。

## `process_shard` 与 `pretokenize`：并行处理 50 个 shard

```python title="tinystories.py -- pretokenize(): 多进程调度" hl=4-5
def pretokenize(vocab_size):
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    fun = partial(process_shard, vocab_size=vocab_size)
    with ProcessPoolExecutor() as executor:
        executor.map(fun, enumerate(shard_filenames))
```

```python title="tinystories.py -- process_shard(): 单个 shard 的真实工作" hl=8
def process_shard(args, vocab_size):
    shard_id, shard = args
    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)
    with open(shard, "r") as f:
        data = json.load(f)
    all_tokens = []
    for example in tqdm(data, position=shard_id):
        text = example["story"].strip()
        tokens = enc.encode(text, bos=True, eos=False)
        all_tokens.extend(tokens)
    all_tokens = np.array(all_tokens, dtype=np.uint16)
```

<div data-diagram="data-shard-pipeline" data-caption="50 个 shard 并行编码，再切成不重叠定长块打乱顺序"></div>

- `functools.partial` 只是"预先绑定参数"，**跟并行无关**；真正的并行来自 `ProcessPoolExecutor`（多进程，绕开 Python GIL，适合 CPU 密集型任务）。
- `enumerate(shard_filenames)` 生成 `(0,path0), (1,path1), ...`——这个编号是 `enumerate` 动态生成的，不是文件名自带的。
- `tqdm(data, position=shard_id)` 里的 `position` 只是控制多个并行进度条各画在终端哪一行，纯 UI 效果，跟数据处理逻辑无关。
- 每条故事编码时 `enc.encode(text, bos=True, eos=False)`——**只加 BOS，不加 EOS**。50 个 `.json` → 50 个 `.bin`，一一对应。

## `PretokDataset`：对应 nanoGPT 的 `get_batch()`

```python title="tinystories.py -- PretokDataset.__iter__()" hl=6,10-11
while True:
    rng.shuffle(shard_filenames)
    for shard in shard_filenames:
        # open the dataset for reading but keep it on disk with memmap
        m = np.memmap(shard, dtype=np.uint16, mode="r")
        num_batches = len(m) // self.max_seq_len
        num_batches -= 1  # drop the last partial batch
        ixs = list(range(num_batches))
        rng.shuffle(ixs)
        for ix in ixs:
            start = ix * self.max_seq_len
            end = start + self.max_seq_len + 1
            # calling .astype will copy the data into a new numpy array, now in RAM
            chunk = torch.from_numpy((m[start:end]).astype(np.int64))
            x = chunk[:-1]
            y = chunk[1:]
            yield x, y
```

| | nanoGPT `get_batch()` | `PretokDataset` |
|---|---|---|
| 采样方式 | 完全随机起点（允许重叠） | 切成不重叠固定块，**打乱块的顺序** |
| 底层存储 | 整个数据集常驻内存 | `np.memmap` 懒加载，只有 `.astype()` 那步才真正拷贝进内存 |

**关键点：`max_seq_len` 切分对 BOS（`<s>`）完全"视而不见"**——纯粹按固定长度机械切片，不关心切出来的 chunk 是不是跨越了故事边界。一个训练样本完全可能是"故事 A 结尾 + BOS + 故事 B 开头"拼在一起，模型要自己从 BOS 这个特殊 token 学会"这里是新上下文的开始"。

## `Task.iter_batches`：Python 工程模式集合

```python title="tinystories.py -- Task.iter_batches()" hl=8-9
class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            yield x, y
```

- **`@staticmethod`**：不需要 `self`，`Task` 是纯粹的"命名空间类"（不是 GoF 工厂模式），从未被实例化，只是把函数组织在一个有意义的名字下。
- **`**dataset_kwargs`**：关键字变长参数，用来**透传**——`iter_batches` 不用把 `PretokDataset` 需要的 `split`/`max_seq_len` 等参数名再抄一遍，直接原样转发，DRY 原则。
- 这里的 `partial` 用途和 `tinystories.py` 里那次不同：那次是"凑参数个数给 `executor.map` 用"；这次纯粹是"把基本不变的配置提前绑好，只留 `split` 每次变化"。
- `DataLoader` 自动把单条 `(T,)` 样本 stack 成 `(B,T)`，`num_workers` 多进程并行预取，`pin_memory=True` 的作用见「训练工程细节」一页。

<div class="aside">
<b>与 nanoGPT 的实质差异：</b>llama2.c 是「切块 + 打乱块顺序」，nanoGPT 是「每步都随机选新起点」——两者都不管故事/文档边界，边界信息只能靠 BOS/EOS 这类特殊 token 让模型自己学，这是数据管道设计上的共同权衡，不是谁更"正确"。
</div>
