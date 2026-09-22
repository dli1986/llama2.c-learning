## `tokenizer.bin` ≠ 训练数据打包——两者完全不同，依赖方向也相反

| | `dataXX.bin`（50 个） | `tokenizer.bin`（1 个） |
|---|---|---|
| 内容 | **训练数据**——故事编码后的 token id 序列 | **词表本身**——每个 id 对应的字符串 + 分数 |
| 产生方式 | `process_shard()` 调用 `enc.encode()` | `tokenizer.py` 的 `export()`，转换自 `tokenizer.model` |
| 依赖关系 | **依赖** `tokenizer.model`（先有词表才能编码） | 不依赖任何 `dataXX.bin` |

```
tokenizer.model(Meta预训练,一份) --export()--> tokenizer.bin(给run.c用)
                ↓ 被 process_shard() 使用
原始故事文本(.json) --encode--> dataXX.bin(50个)
```

## Python 侧：`Tokenizer.encode()` 只是薄包装，不是重新实现 BPE

```python title="tokenizer.py -- Tokenizer.encode()" hl=2
def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
    t = self.sp_model.encode(s)
    if bos:
        t = [self.bos_id] + t
    if eos:
        t = t + [self.eos_id]
    return t
```

包一层的目的是让 Python 端和 `run.c` 手写的 C 版本 `encode(bos, eos, ...)` **接口一致**，不是 SentencePiece 缺功能。真正的 BPE 分词逻辑完全在 `sp_model.encode(s)` 内部（高亮那一行），Python 这边只是加了 BOS/EOS 的前后缀逻辑。

## C 侧：`run.c` 从零手写 BPE 贪心合并循环

`run.c` 没有依赖 SentencePiece 的 C++ 库，而是自己重新实现了一遍 BPE：先把原始 UTF-8 字节流查表变成单字节 token，再不断合并分数最高的相邻 token 对，直到无法再合并——这是 `Tokenizer.encode()` 里 `self.sp_model.encode(s)` 一行背后，SentencePiece 真正在做的事情，`run.c` 把它原样重写了一遍：

```c title="run.c -- encode(): BPE 贪心合并主循环" hl=8-9,17
// merge the best consecutive pair each iteration, according the scores in vocab_scores
while (1) {
    float best_score = -1e10;
    int best_id = -1;
    int best_idx = -1;

    for (int i=0; i < (*n_tokens-1); i++) {
        // check if we can merge the pair (tokens[i], tokens[i+1])
        sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1 && t->vocab_scores[id] > best_score) {
            // this merge pair exists in vocab! record its score and position
            best_score = t->vocab_scores[id];
            best_id = id;
            best_idx = i;
        }
    }

    if (best_idx == -1) {
        break; // we couldn't find any more pairs to merge, so we're done
    }

    // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
    tokens[best_idx] = best_id;
    // delete token at position best_idx+1, shift the entire sequence back 1
    for (int i = best_idx+1; i < (*n_tokens-1); i++) {
        tokens[i] = tokens[i+1];
    }
    (*n_tokens)--; // token length decreased
}
```

每一轮 `for` 循环都要重新扫描**当前所有**相邻 token 对，把它们两两拼接成字符串（`sprintf(str_buffer, "%s%s", ...)`），再用 `str_lookup`（对排序好的词表做二分查找）确认这个拼接结果是不是词表里的一个合法 token；如果是，比较 `vocab_scores` 选出分数最高的一对（第 8-9 行），全部扫完后合并这一对（`tokens[best_idx] = best_id`），后面的 token 整体前移一位，`n_tokens` 减一，然后**从头再来**——直到某一轮扫描不到任何可合并的对（`best_idx == -1`，第 17 行）才停止。这是 $O(n^2)$ 量级的朴素实现（每轮都要重新扫一遍），换来的是零依赖、几十行 C 就能复现 SentencePiece 的合并逻辑。

**为什么每个 token 都需要一个"分数"（score）**——这不是 Llama 专属设计，而是 BPE 贪心合并算法的**结构性要求**：每一轮要在多个"可合并的相邻 token 对"里选出一个优先合并，没有分数就没法裁决优先级，算法根本跑不起来。不同实现里分数的具体计算方式（频率、log 概率、合并顺序）可能不同，但"每个 token 必须有优先级"这件事是共通的。

## `export()`：byte-for-byte 布局必须与 `run.c` 的 `build_tokenizer` 严格对齐

真正写文件的 Python 代码——`struct.pack` 的字段顺序（`I` = unsigned int，`f` = float）必须跟 C 端 `fread` 的顺序逐字节对上：

```python title="tokenizer.py -- Tokenizer.export()" hl=13-15
def export(self):
    tokens, scores = [], []
    for i in range(self.n_words):
        t = self.sp_model.id_to_piece(i)
        s = self.sp_model.get_score(i)
        if i == self.bos_id:
            t = '\n<s>\n'
        elif i == self.eos_id:
            t = '\n</s>\n'
        t = t.replace('\u2581', ' ')  # sentencepiece uses this character as whitespace
        b = t.encode('utf-8')
        tokens.append(b)
        scores.append(s)

    max_token_length = max(len(t) for t in tokens)
    tokenizer_bin = self.model_path.replace('.model', '.bin')
    with open(tokenizer_bin, 'wb') as f:
        f.write(struct.pack("I", max_token_length))
        for bytes, score in zip(tokens, scores):
            f.write(struct.pack("fI", score, len(bytes)))
            f.write(bytes)
```

C 端严格对称地按同一顺序读回来（`build_tokenizer`，只读一次 `max_token_length`，然后逐条读 `score → len → 内容`）：

```c title="run.c -- build_tokenizer(): 严格顺序读取，不能跳跃" hl=3-7
if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { ... }
int len;
for (int i = 0; i < vocab_size; i++) {
    fread(t->vocab_scores + i, sizeof(float), 1, file);
    fread(&len, sizeof(int), 1, file);
    t->vocab[i] = (char *)malloc(len + 1);
    fread(t->vocab[i], len, 1, file);
    t->vocab[i][len] = '\0';
}
```

<div data-diagram="tokenizer-binary-format" data-caption="变长记录格式：只能顺序扫描，不能像权重那样用乘法直接跳到任意位置"></div>

这是**变长记录格式**，跟权重文件那种固定 stride 的布局完全不同——`run.c` 必须**顺序**读，不能像读权重那样用 `l*dim*dim` 这类乘法公式直接跳到任意 token 的位置，因为每个 token 的字节内容长度都不一样（每次读 `len` 之前，压根不知道下一条record 有多长）。

## BOS/EOS：保留的特殊哨兵 id，不是文本内容

- 跟 `\n`/`\r` 类比：角色相似（都当分隔符用），但机制不同——`\n` 是**内容**（文本里本来就有的普通字符），BOS/EOS 是**词表里专门保留、不对应任何真实文字的 id**，正常文本编码不会自然产生它们，只能被 `encode()` 人为插入。
- **不同词表的 BOS/EOS id 不同**（Llama 2 是 `<s>`=1，`</s>`=2），这是每个 tokenizer 自己的设计，没有统一标准。权威来源是 `sp_model.bos_id()` 这个 API 查询（读的是 `.model` 文件内嵌的元数据），不是外部文档。
- **实际用途**：在拼接后完全同质的 token id 流里充当"故事边界标记"——`process_shard` 给每条故事开头加 BOS（不加 EOS），后续 `avg_seq_len = all_tokens.size / (all_tokens==1).sum()` 就是靠数 "1"（BOS id）出现的次数反推有多少条故事。

<div class="aside evidence">
<b>证据链：</b>整套 tokenizer 二进制格式的对齐关系是「有代码可查」而非推测——`tokenizer.py` 的 <code>export()</code> 与 <code>run.c</code> 的 <code>build_tokenizer</code> 必须读写同一份字节顺序，这也是为什么变长记录格式在 C 端必须严格顺序解析。
</div>
