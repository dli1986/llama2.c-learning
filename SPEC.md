# llama2c-site — Specification

## Purpose

A static, mobile-friendly companion site walking through Karpathy's
[llama2.c](https://github.com/karpathy/llama2.c), built from the author's
own real learning notes rather than re-summarizing the upstream README.
Sibling project to `nanoGPT/nanogpt-site/` — same visual language, same
"one topic per page" philosophy, but a different source pipeline since
there's no notebook to parse cells from here.

## Source of truth

`/home/dli/llama2.c/STUDY_NOTES.md` (WSL Ubuntu, 757 lines as of
2026-09-22) — a running set of notes accumulated while reading `run.c`
against `train.py`/`model.py`/`tinystories.py`/`tokenizer.py` and comparing
with nanoGPT's `gpt-dev.ipynb`/`bigram.py`. The notes are mid-way through
(not yet covering `run.c`'s sampling internals, `train.py`'s LR
scheduler/optimizer grouping, or PyTorch autograd) — the site's last page
("学习进度与下一步") states this explicitly rather than fabricating
coverage of unstudied material.

This is **not** a live-generated site (unlike nanogpt-site, which parses
`gpt-dev.ipynb` fresh every build). Content pages in `content/pages/*.md`
are hand-authored/restructured from STUDY_NOTES.md's accumulated Q&A order
into a pedagogical one-topic-per-page order — no facts beyond what the
notes contain were introduced.

## Page reorganization rationale

STUDY_NOTES.md accumulates in the order topics came up during study
sessions (tokenizer details appear after model internals, RoPE math is a
follow-up to an earlier RoPE mention, etc.). The site reorders into a
cleaner progression matching the four angles the site is meant to
emphasize (Python/C, PyTorch internals, memory layout, real engineering
tradeoffs):

1. Overview (pipeline) → 2. Tokenizer → 3. Data pipeline → 4. PyTorch model
internals (`nn.Module`) → 5. Architecture diffs (RMSNorm/RoPE/SwiGLU/GQA)
→ 6. RoPE math deep-dive → 7. C tensor/memory layout → 8. RunState vs
Weights → 9. `run.c` engineering (mmap, prefill) → 10. Training engineering
(`pin_memory`, `configurator.py`) → 11. Full nanoGPT-vs-llama2.c comparison
table → 12. Honest learning-status page.

## Diagram technology: hand-authored SVG, no canvas

Unlike `nanogpt-site` (which had to retarget legacy Canvas2D drawing code
via a `canvas2svg.js` shim — see that project's SPEC.md), there is no
legacy diagram code here to port: every diagram is authored directly as
static SVG markup, applying the lesson learned from that earlier project
(see `nanogpt-site.md` repo memory notes): narrow intrinsic `viewBox`
(~380-440px wide) so 100%-width mobile scaling keeps text near-native
size, one concept per diagram rather than dense multi-panel desktop
layouts, and no client-side JS dependency for rendering.

11 diagrams in `static/svg/`, each inlined into its page via a
`<div data-diagram="slug"></div>` marker in the page's Markdown (a raw
HTML block that `markdown-it-py`'s `html: True` option passes through
unchanged; `build.py` then regex-replaces it with the actual SVG file
contents wrapped in `.diagram-svg-wrap`).

## Tech stack

- **Build**: Python (`MyTest` venv — Jinja2, markdown-it-py, Pygments,
  PyYAML, all already installed for nanogpt-site, nothing new added).
- **Templating**: Jinja2 (`base.html.jinja` + `page.html.jinja` +
  `index.html.jinja`).
- **Markdown**: `markdown-it-py` (CommonMark + GFM tables enabled,
  `html: True` for the diagram-marker passthrough).
- **Syntax highlighting**: Pygments (`monokai` style), Python/C lexers
  selected by the fenced code block's language tag.
- **Math**: KaTeX via CDN (`auto-render`), client-side, `$...$`/`$$...$$`
  delimiters — the only external network dependency the site has; a
  deliberate tradeoff for correct-looking math without a server-side
  LaTeX-to-HTML pipeline (see README for the tradeoff note).
- **Serving**: `server.py`, stdlib `http.server`, binds `0.0.0.0`.

## Directory layout

```
llama2/llama2c-site/
  SPEC.md, README.md
  content/
    pages.yaml              page order/titles/nav (no prose here)
    pages/NN-slug.md         hand-authored prose per page
  templates/
    base.html.jinja, page.html.jinja, index.html.jinja
  static/
    css/style.css            dark theme adapted from nanogpt-site
    svg/*.svg                 11 hand-authored diagrams
  build/build.py               entry point: content/pages.yaml + *.md ->
                                 dist/
  dist/                        generated output (git-ignored)
  server.py                    local/LAN static file server
```

## Deployment

Same pattern as `nanogpt-learning` in `Duo-digital-garden/` (see that
project's memory notes): `dist/*` gets copied into
`Duo-digital-garden/public/llama2c-learning/`, linked from a project MDX
entry via `demoUrl: /llama2c-learning/index.html` (explicit filename, not
a bare directory path — bare directory paths break the site's relative
static asset links, a bug already hit and fixed once for nanogpt-learning).

## Status

All 12 pages + 11 diagrams built and verified via a local build run. Known
gap (see page 12, "学习进度与下一步"): `run.c` sampling internals,
`train.py`'s LR scheduler/optimizer-grouping details, and PyTorch autograd
itself are intentionally NOT covered — the study notes hadn't reached
those yet as of this site's initial build.
