"""
phosphorus.dsl.backtick
~~~~~~~~~~~~~~~~~~~~~~~
Ultra‑light DSL for `PhiValue` literals in IPython / Colab.

* Type a single back‑tick `` ` `` followed by any Python expression.
* A token transformer replaces that back‑tick with ``lambda φ:``.
* An AST transformer turns ``lambda φ: BODY`` into ``PhiValue(<AST of BODY>)``.
* Alternate syntax using subscript notation: ``φ[EXPR]``, ``phi[EXPR]``, ``qq[EXPR]`` etc

Why it works
============
`lambda` is very low precedence, so the *longest* following expression is
captured as the body; no closing delimiter is needed.
"""

import ast
import tokenize
from dataclasses import dataclass
from io import StringIO
from typing import Iterable

from IPython import get_ipython
from p4s.core.phivalue import PhiValue

# TODO: (`beta(A.e)).stype throws an error somehow

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

PHI_NAME = "φ"   # single arg name for injected lambda
BACKTICK = "`"  # trigger character

@dataclass(frozen=True, slots=True)
class PhiSubscriptPrefix:
  """Lightweight marker object for phi-style DSL prefixes (e.g., φ[EXPR])."""
  name: str

  def __repr__(self) -> str:
    return f"PhiSubscriptPrefix(name={self.name!r})"

phi = PhiSubscriptPrefix("phi")
ϕ = PhiSubscriptPrefix("ϕ")
qq = PhiSubscriptPrefix("qq")
φ = PhiSubscriptPrefix("φ")
ɸ = PhiSubscriptPrefix("ɸ")

PHI_SUBSCRIPT_PREFIXES = {p.name: p for p in (phi, ϕ, qq, φ, ɸ)}
PHI_SUBSCRIPT_NAMES = set(PHI_SUBSCRIPT_PREFIXES)

__all__ = [
  "PHI_SUBSCRIPT_NAMES",
  "PhiSubscriptPrefix",
  "phi",
  "ϕ",
  "qq",
  "φ",
  "ɸ",
]

# ---------------------------------------------------------------------------
# 1. token‑level transformer  –   `` ` `` → tokens for ``lambda φ:``
# ---------------------------------------------------------------------------

# We use the *tuple* form expected by `tokenize.untokenize`, so we don’t
# need to compute start/end columns.  The tuple is just (tok_type, tok_str).

LAMBDA_SEQ = [
  (tokenize.NAME, "lambda"),
  (tokenize.NAME, PHI_NAME),
  (tokenize.OP,   ":"),
]

def backtick_token_transform(tokens: Iterable[tokenize.TokenInfo]):
  """Replace every back‑tick token with the tuple sequence for `lambda φ:`."""
  for tok in tokens:
    if tok.string == BACKTICK:
      for part in LAMBDA_SEQ:
        yield part
    else:
      yield (tok.type, tok.string)


# ---------------------------------------------------------------------------
# 2. AST transformer – lambda φ: BODY and subscripts →  PhiValue(<AST of BODY>)
# ---------------------------------------------------------------------------

class PhiValueASTTransformer(ast.NodeTransformer):
  """Rewrite phi-lambda and phi-like subscripts into PhiValue literals."""

  def visit_Lambda(self, node: ast.Lambda):  # noqa: N802
    # match exactly one arg named φ
    if len(node.args.args) == 1 and node.args.args[0].arg == PHI_NAME:
      # unparse and re-parse the body to get a fresh AST node
      body_src = ast.unparse(node.body)
      new_call = ast.Call(
        func=ast.Name(id="PhiValue", ctx=ast.Load()),
        args=[ast.Constant(value=body_src)],
        keywords=[],
      )
      return ast.copy_location(new_call, node)
    return self.generic_visit(node)

  def visit_Subscript(self, node: ast.Subscript):  # noqa: N802
    if isinstance(node.value, ast.Name) and node.value.id in PHI_SUBSCRIPT_NAMES:
      slice_src = ast.unparse(node.slice)
      new_call = ast.Call(
        func=ast.Name(id="PhiValue", ctx=ast.Load()),
        args=[ast.Constant(value=slice_src)],
        keywords=[],
      )
      return ast.copy_location(new_call, node)
    return self.generic_visit(node)

# ---------------------------------------------------------------------------
# 3. installer – no-op outside IPython
# ---------------------------------------------------------------------------

def install_backtick_dsl() -> None:
  """Register token & AST transformers in IPython/Colab."""
  ip = get_ipython()
  if ip is None:
    return

  # token transformer
  def token_replacer(code):
    text = code
    if isinstance(code, list):
      text = ''.join(code)
    toks = tokenize.generate_tokens(StringIO(text).readline)
    replaced = tokenize.untokenize(backtick_token_transform(toks))
    return replaced.splitlines(keepends=True) if isinstance(code, list) else replaced
  ip.input_transformers_post.append(token_replacer)

  # AST transformer
  ip.ast_transformers.append(PhiValueASTTransformer())

# auto‑install in notebooks
if get_ipython() is not None:
  install_backtick_dsl()
