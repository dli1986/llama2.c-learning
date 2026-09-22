## `pin_memory`/`non_blocking`：不是公式，是真实的硬件约束

<div data-diagram="pin-memory-dma" data-caption="不锁页：静默退化为同步拷贝；锁页：真异步，可与 GPU 计算重叠"></div>

**根源问题**：普通（可分页）内存的虚拟地址→物理地址映射，操作系统随时可能改变（swap、内存整理）；GPU 的 DMA 引擎直接按物理地址读写，不经过 CPU——如果传输过程中物理地址被换出/挪动，DMA 会读到错误数据。

**"pinned"/锁页内存的不变量**：虚拟地址到物理地址的映射**冻结不变**。这是 DMA 能安全直接读写这块内存的前提，不是性能优化技巧，是正确性要求。

**不锁页会发生什么——CUDA 驱动偷偷做了一次隐藏的同步拷贝**：

```
普通内存 → CUDA驱动先同步拷贝到内部锁页中转缓冲区(阻塞!) → 再异步DMA到显存
```

`non_blocking=True` 如果源内存不是锁页的，会**静默退化**成同步拷贝，不报错，只是"non_blocking"这个提示不生效。

**`DataLoader(pin_memory=True)` 的作用**：worker 进程产出的 tensor 默认是普通内存，`pin_memory=True` 让 `DataLoader` 在交给主进程前，额外拷贝一次到锁页内存——用一次确定的 CPU 开销，换取后面 `.to(device, non_blocking=True)` 能**真正异步、能与 GPU 计算重叠**的资格。

**具体重叠点**（`train.py` 训练主循环的真实结构）：

```python title="train.py -- 训练循环: 异步预取与反向传播重叠" hl=5
for micro_step in range(gradient_accumulation_steps):
    with ctx:
        logits = model(X, Y)
        loss = raw_model.last_loss
        loss = loss / gradient_accumulation_steps
    # immediately async prefetch next batch while model is doing the forward pass on the GPU
    X, Y = next(train_batch_iter)
    scaler.scale(loss).backward()
```

意图：GPU 算当前 batch（`model(X, Y)`）的同时，CPU 应该已经在准备+搬运下一个 batch（第 5 行 `next(train_batch_iter)` 紧跟在 forward 之后、backward 之前调用）——这个"同时"能否兑现，完全取决于 `pin_memory` 有没有生效。

## `configurator.py`：没有 `main()`/`argparse`，靠 `exec` 就地改全局变量

**Karpathy 的原话——不是简单的"不喜欢"，是"明知丑陋但主动取舍"**：

```python title="configurator.py -- 模块开头的自白"
"""
Poor Man's Configurator. Probably a terrible idea.
...
I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable.
"""
```

他自己承认这大概率是个"糟糕的主意"，但**更讨厌"每次访问配置都要写 `config.xxx` 前缀"这件事**，主动选择了这个 hack，用架构上的不干净换取 `train.py` 正文里所有超参数都能直接裸写 `batch_size` 而不是 `config.batch_size`。**不是为了"模块化、干净"**——如果追求模块化/干净，用 `argparse`+`dataclass` 反而更合适；这里牺牲的恰恰是配置机制本身的规范性。

**机制拆解**：

```python title="train.py -- config_keys 扫描 + exec() 注入" hl=3
config_keys = [k for k, v in globals().items()
               if not k.startswith("_") and isinstance(v, (int, float, bool, str))]
exec(open("configurator.py").read())
config = {k: globals()[k] for k in config_keys}
```

- `config_keys`：扫一遍 `train.py` 当前的全局变量，挑出"简单类型"的那些（即开头那一堆超参数），排除函数/模块等复杂对象，留作后面日志记录用。
- **`exec()` 重要纠正**：Python 内建函数 `exec()` 跟 C/POSIX 的 `exec` 系列系统调用（`execve` 等）**完全不是一回事**，只是同名。POSIX 的 `exec` 是"用新程序镜像替换当前进程"，涉及进程/内存镜像层面的操作；Python 的 `exec()` **不启动任何新进程或线程**，只是把一段字符串当 Python 代码，在当前解释器、当前命名空间里**当场编译执行**——效果上约等于把这段代码原样写在调用处，但机制上是"运行时动态执行一段字符串"，不是 C 的 `#include` 那种编译前文本替换，也绝非进程级操作。
- 因为这次 `exec()` 没有传独立命名空间，它默认就在 `train.py` 自己的全局作用域里跑，所以 `configurator.py` 内部任何 `globals()[key] = value`，改的就是 `train.py` 自己的全局变量，如同代码被直接拼贴在 `exec` 那一行。

**`configurator.py` 内部：`sys.argv` 解析**：

```python title="configurator.py -- sys.argv 解析" hl=8-9
for arg in sys.argv[1:]:
    if '=' not in arg:
        # assume it's the name of a config file
        exec(open(arg).read())
    else:
        # assume it's a --key=value argument
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            attempt = literal_eval(val)
            assert type(attempt) == type(globals()[key])
            globals()[key] = attempt
```

- `sys.argv` 类似 C 的 `argv[]`（字符串列表），但没有单独的 `argc`（Python 列表自带 `len()`）。
- **`sys.argv[0]` 是脚本文件名 `"train.py"`，不是 `"python"` 本身**——`python` 解释器不会出现在 `argv` 里，道理类似 C 里"argv[0] 是被执行的程序自己"。
- `python train.py --batch_size=256` 时，`sys.argv[1:]` 就是 `["--batch_size=256"]`，被解析成 `key="batch_size", val="256"`，覆盖 `train.py` 里预设的 `batch_size = 128`。

<div class="aside warn">
<b>实测印证这套方案的脆弱性：</b><code>python train.py -h</code> 会崩溃（exit code 1）——因为没有 <code>argparse</code> 那种 <code>--help</code> 处理，<code>-h</code> 不含 <code>=</code>，被当成配置文件路径，<code>open("-h")</code> 找不到文件直接抛异常。这正是 Karpathy 自己承认的"丑陋"的具体表现之一。
</div>
