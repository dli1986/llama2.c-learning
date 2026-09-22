#!/usr/bin/env python3
"""
llama2c-site build script. Lives at llama2/llama2c-site/.

Reads:
  - content/pages.yaml        (page order/metadata)
  - content/pages/*.md        (hand-authored prose, grounded in the author's
                                own STUDY_NOTES.md at /home/dli/llama2.c/,
                                WSL — not copied verbatim, restructured into
                                one-topic-per-page but no facts invented)
  - static/svg/*.svg          (hand-authored diagrams, inlined into pages via
                                a `<div data-diagram="slug"></div>` marker)

Writes:
  - dist/index.html
  - dist/01-overview.html .. dist/12-learning-status.html
  - dist/static/ (copied from static/)

Markdown: markdown-it-py (GFM tables enabled) + Pygments fence highlighting
for ```python/```c blocks. Math: left as literal $...$/$$...$$ in the
rendered HTML, rendered client-side by KaTeX auto-render (see base template)
— no server-side math dependency needed.
"""
import html as html_lib
import re
import shutil
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.lexers import PythonLexer, CLexer, TextLexer
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent.parent
PAGES_YAML = ROOT / "content" / "pages.yaml"
PAGES_DIR = ROOT / "content" / "pages"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"
SVG_DIR = STATIC_DIR / "svg"
DIST_DIR = ROOT / "dist"

pygments_formatter = HtmlFormatter(style="monokai", cssclass="hl-code", nowrap=False)
LEXERS = {
    "python": PythonLexer(),
    "c": CLexer(),
    "text": TextLexer(),
}


def highlight_code(code, lang, attrs):
    lexer = LEXERS.get((lang or "").lower(), TextLexer())
    return highlight(code, lexer, pygments_formatter)


md = MarkdownIt("commonmark", {"html": True, "highlight": highlight_code}).enable("table")

DIAGRAM_RE = re.compile(r'<div data-diagram="([\w-]+)"(?:\s+data-caption="([^"]*)")?\s*></div>')

# CommonMark's backslash-escape rule (backslash + ASCII punctuation -> literal
# escaped char) silently eats one backslash out of LaTeX's "\\" row separator
# (e.g. matrix definitions), corrupting math before KaTeX ever sees it. Fix:
# extract $$...$$/$...$ spans BEFORE markdown-it runs, substitute back the
# raw (HTML-escaped) LaTeX source after rendering.
MATH_BLOCK_RE = re.compile(r'\$\$(.+?)\$\$', re.DOTALL)
MATH_INLINE_RE = re.compile(r'(?<!\$)\$([^$\n]+?)\$(?!\$)')
MATH_PLACEHOLDER_RE = re.compile(r'\uE000MATH(\d+)\uE000')


def protect_math(source):
    store = []

    def block_repl(m):
        store.append(("$$", m.group(1)))
        return f"\uE000MATH{len(store) - 1}\uE000"

    source = MATH_BLOCK_RE.sub(block_repl, source)

    def inline_repl(m):
        store.append(("$", m.group(1)))
        return f"\uE000MATH{len(store) - 1}\uE000"

    return MATH_INLINE_RE.sub(inline_repl, source), store


def restore_math(html, store):
    def repl(m):
        delim, content = store[int(m.group(1))]
        return f"{delim}{html_lib.escape(content, quote=False)}{delim}"

    return MATH_PLACEHOLDER_RE.sub(repl, html)


def inline_diagrams(html):
    def repl(m):
        slug, caption = m.group(1), m.group(2) or ""
        svg_path = SVG_DIR / f"{slug}.svg"
        if not svg_path.exists():
            return f'<div class="diagram-placeholder">[missing diagram: {slug}]</div>'
        svg = svg_path.read_text(encoding="utf-8")
        cap_html = f'<div class="diagram-caption">{caption}</div>' if caption else ""
        return f'<div class="diagram-section"><div class="diagram-svg-wrap">{svg}</div>{cap_html}</div>'
    return DIAGRAM_RE.sub(repl, html)


def load_pages():
    with open(PAGES_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["pages"]


def main():
    pages = load_pages()
    nav = [{"num": p["num"], "slug": p["slug"], "title": p["title"], "subtitle": p.get("subtitle", "")} for p in pages]

    if DIST_DIR.exists():
        shutil.rmtree(DIST_DIR)
    DIST_DIR.mkdir(parents=True)
    shutil.copytree(STATIC_DIR, DIST_DIR / "static")

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    page_template = env.get_template("page.html.jinja")
    index_template = env.get_template("index.html.jinja")

    common_ctx = {"nav": nav, "asset_version": "1"}

    (DIST_DIR / "index.html").write_text(
        index_template.render(**common_ctx), encoding="utf-8"
    )

    for i, p in enumerate(pages):
        md_path = PAGES_DIR / f"{p['slug']}.md"
        source = md_path.read_text(encoding="utf-8")
        protected, math_store = protect_math(source)
        body_html = inline_diagrams(restore_math(md.render(protected), math_store))
        prev_page = pages[i - 1] if i > 0 else None
        next_page = pages[i + 1] if i < len(pages) - 1 else None
        html = page_template.render(
            page=p, body_html=body_html, num=p["num"], total=len(pages),
            prev=prev_page, next=next_page, **common_ctx
        )
        (DIST_DIR / f"{p['slug']}.html").write_text(html, encoding="utf-8")

    print(f"Built {len(pages)} pages + index -> {DIST_DIR}")


if __name__ == "__main__":
    main()
