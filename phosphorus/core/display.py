# phosphorus/core/display.py

from __future__ import annotations
import html
import ast

from typing import Literal
from black import format_str, Mode
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import HtmlFormatter


from IPython.display import HTML, display

_CSS = """
<style id="phi-css">
  .phi-wrapper {
    display: grid;
    grid-gap: .4em;
    grid-auto-flow: column;
    grid-template-columns: minmax(60ch, max-content) auto; /* ≥60ch, but grow to fit code, then badge */
    grid-gap: .4em;
    align-items: start;
    justify-items: start;
    padding: .4em .6em;
    border-radius: 6px;
    background: var(--jp-layout-color1, #f5f5f5);
    color: var(--jp-ui-font-color1, #000);
  }
  @media (prefers-color-scheme: dark) {
    .phi-wrapper {
      background: var(--jp-layout-color0, #2b2b2b);
      color: var(--jp-ui-font-color0, #eee);
    }
  }
  .phi-code  { white-space: pre-wrap; margin: 0 }
  .phi-badge {
      display: inline-block;
      font-weight: bold;
      padding: .15em .4em;
      border-radius: 4px;
      background: #c8c8ff;
  }
</style>
"""

# Module-level flag to ensure CSS is injected only once
_css_injected: bool = False

def inject_css():
  """Call this **once** (e.g. at package import) so every widget shares it."""
  global _css_injected
  if _css_injected:
      return
  # Display the CSS block exactly once
  display(HTML(_CSS))
  _css_injected = True

# ---------- low‑level HTML makers ---------- #
def make_code_html(code: str, *, font_size: str) -> str:
  """Monospace, syntax‑highlighted <pre> – no badge."""
  highlighted = highlight(
    code,
    PythonLexer(),
    HtmlFormatter(noclasses=True, nowrap=True),
  )
  return (
    f"<pre class='phi-code' style='font-size:{font_size};"
    f"font-family:var(--jp-code-font-family,monospace);'>"
    f"{highlighted}</pre>"
  )


def make_badge_html(stype, *, font_size: str) -> str:
  if stype and getattr(stype, 'is_unknown', False):
    stype = None
  if not stype:
    return ''
  txt = html.escape(repr(stype)) if stype else ''
  return (
    f"<span class='phi-badge' style='font-size:{font_size};'>"
    f"{txt}</span>"
  )
# ------------------------------------------ #

def render_phi_html(
  code: str | ast.AST,
  stype: object = None,
  *,
  layout: Literal['inline', 'stacked'] = 'inline',
  line_length: int = 78,
  font_size: str = '14px',
) -> str:
  """
  Produce an HTML snippet for a PhiValue.

  layout = 'inline'   →  code + badge on the same row
  layout = 'stacked'  →  badge on 1st row, code below (compact for tree nodes)
  """

  # 1) normalise inputs
  if isinstance(code, ast.AST):
    code = ast.unparse(code)

  # 2) pretty‑print via Black
  pretty = format_str(code, mode=Mode(line_length=line_length))

  # 3) build the sub‑blocks
  code_block  = make_code_html(pretty, font_size=font_size)
  badge_block = make_badge_html(stype, font_size=font_size)

  # 4) compute a reasonable min‑width based on longest line of `pretty`
  max_chars = max(len(line) for line in pretty.splitlines())
  width_css = f"min-width:{max_chars + 4}ch;"   # +4 for padding
  width_css = ''

  # 5) assemble according to layout
  if layout == 'inline':
    inner = f"{code_block}{badge_block}"
    flow  = "column"
  else:  # stacked
    inner = f"{code_block}{badge_block}"
    flow  = "row"

  return (
    f"{_CSS}"
    f"<div class='phi-wrapper' style='grid-auto-flow:{flow};{width_css}'>"
    f"{inner}</div>"
  )

def xrender_phi_html(code: str | ast.AST, stype: object, guard: str | ast.AST) -> str:
  """
  Given a code string (or AST) and semantic type, produce a self-contained HTML
  snippet that preserves Black's line breaks, applies syntax highlighting,
  and displays the type badge to the right of the longest code line.

  Adapts automatically to light/dark themes in Jupyter/Colab via CSS variables.
  """
  # 0) Unparse if an AST was passed
  if isinstance(code, ast.AST):
    code = ast.unparse(code)
  if isinstance(guard, ast.AST):
    guard = ast.unparse(guard)

  # 1) Auto-format code using Black
  pretty = format_str(code, mode=Mode())

  # 2) Syntax-highlight via Pygments (inline CSS, no external classes)
  highlighted = highlight(
    pretty,
    PythonLexer(),
    HtmlFormatter(noclasses=True, nowrap=True),
  )

  # 3) Wrap code in <pre> so line breaks and indenting are exact, but allow wrap
  code_html = (
    f"<pre style='margin:0; white-space:pre-wrap;"
      f"font-family:var(--jp-code-font-family,monospace);'>"
    f"{highlighted}</pre>"
  )

  # 4) Prepare the type badge
  if stype and stype.is_unknown:
    stype = None
  badge = (
    f"<span style='display:inline-block;"
      f"font-family:var(--jp-code-font-family,monospace);"
      f"font-weight:bold; padding:0.15em 0.4em; margin-left:0.8em;"
      f"border-radius:4px; background-color:#c8c8ff; color:#000;'>"
    f"{html.escape(repr(stype))}" + 
    (f" | {html.escape(str(guard))}" if guard else '') +
    "</span>"
  )

  # 5) Return wrapper with table layout for alignment and theme support
  return f"""<div>
  <style>
    .pv-wrapper {{
      display: table;
      padding: 0.5em;
      border-radius: 6px;
      background: var(--jp-layout-color1, #f5f5f5);
      color: var(--jp-ui-font-color1, #000);
    }}
    @media (prefers-color-scheme: dark) {{
      .pv-wrapper {{
        background: var(--jp-layout-color0, #2b2b2b);
        color: var(--jp-ui-font-color0, #eee);
      }}
    }}
    .pv-code, .pv-badge {{ display: table-cell; vertical-align: top; }}
    .pv-code {{
      padding-right: 1em;
      text-align: left;
    }}
    .pv-badge {{ padding-left: 0.5em; }}
  </style>
  <div class="pv-wrapper">
    <div class="pv-code">{code_html}</div>
    <div class="pv-badge">{badge}</div>
  </div></div>
  """