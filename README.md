# llama2c-site

A static, mobile-friendly walkthrough of Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c), companion to
[`nanoGPT/nanogpt-site/`](../../nanoGPT/nanogpt-site/). One page per topic,
grounded entirely in the author's own real study notes
(`/home/dli/llama2.c/STUDY_NOTES.md`, WSL Ubuntu) — content is
hand-authored/restructured from that source, not machine-generated or
invented, and the site openly documents which parts of `llama2.c` haven't
been studied in depth yet (see the last page, "学习进度与下一步").

## Requirements

Uses the existing `MyTest` Python venv (already has Jinja2, Pygments,
markdown-it-py, PyYAML installed — nothing new to install):

```
C:\Users\dli\Projects\MyTest\MyTest\Scripts\python.exe
```

## Build the site

```powershell
cd C:\Users\dli\Projects\MyTest\llama2\llama2c-site
C:\Users\dli\Projects\MyTest\MyTest\Scripts\python.exe build\build.py
```

Regenerates `dist/` from `content/pages.yaml` + `content/pages/*.md` +
`static/`. Re-run after editing any content page or diagram.

## Serve it locally

```powershell
C:\Users\dli\Projects\MyTest\MyTest\Scripts\python.exe server.py
```

Binds `0.0.0.0:8000` — reachable from this machine or your phone on the
same Wi-Fi/LAN.

## Project layout

```
content/pages.yaml      page order/titles/nav metadata
content/pages/*.md       hand-authored prose per page (12 pages)
templates/               Jinja2 templates (base + page + index)
static/css/style.css     dark theme (adapted from nanogpt-site)
static/svg/*.svg         hand-authored diagrams (11), narrow viewBox for
                         mobile — inlined into pages via
                         <div data-diagram="slug"></div> markers
build/build.py           the generator — reads sources, writes dist/
dist/                    generated static site (git-ignored, rebuild anytime)
server.py                trivial static file server
```

See [`SPEC.md`](./SPEC.md) for the full content map and design rationale.

## Status

All 12 pages + 11 diagrams built. Content covers: full pipeline overview,
tokenizer (Python wrapper vs C from-scratch BPE), training data pipeline,
`nn.Module` registration internals, architecture differences vs nanoGPT
(RMSNorm/RoPE/SwiGLU/GQA), RoPE math derivation, C-side tensor memory
layout, RunState vs TransformerWeights, `run.c` engineering tradeoffs
(mmap/no batched prefill), training engineering (`pin_memory`,
`configurator.py`), the full nanoGPT-vs-llama2.c comparison table, and an
honest "not yet covered" status page.

Deployed copy lives at `Duo-digital-garden/public/llama2c-learning/` (same
pattern as `nanogpt-learning`) — re-copy `dist/` there after any rebuild.
