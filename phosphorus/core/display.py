# phosphorus/core/display.py

from __future__ import annotations
import html
import ast

from black import format_str, Mode
from pygments import highlight
from pygments.lexers.python import PythonLexer
from pygments.formatters import HtmlFormatter


def render_phi_html(code: str | ast.AST, stype: object, guard: str | ast.AST) -> str:
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
  return f"""
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
      min-width: 45ch;
    }}
    .pv-badge {{ padding-left: 0.5em; }}
  </style>
  <div class="pv-wrapper">
    <div class="pv-code">{code_html}</div>
    <div class="pv-badge">{badge}</div>
  </div>
  """