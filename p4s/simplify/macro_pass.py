# phosphorus/simplify/macro_pass.py
"""MacroExpander: inlines function/method calls whose *parameters* are
annotated with ``ast.AST``.  Any argument corresponding to such a
parameter is passed *as the raw AST*; all other arguments are evaluated
in the current environment.  The call is replaced by the AST returned by
that function (if it returns an AST)."""
from __future__ import annotations

import ast
import inspect
from typing import Dict, List, Any, get_type_hints

from .passes import SimplifyPass  # base class provides env

# ---------------------------------------------------------------------------
# Helper: evaluate an expression AST under a given env (globals+locals)
# ---------------------------------------------------------------------------

def _eval_ast(expr: ast.AST, env: Dict[str, Any]) -> Any:
  """Safely evaluate *expr* in *env* (LEGB ChainMap)."""
  code = compile(ast.Expression(expr), filename="<ast>", mode="eval")
  return eval(code, env)


def macro_add(l: ast.AST, r: Any) -> ast.AST:
  """Example macro: add literal 'r' to AST 'l'."""
  return ast.BinOp(left=l, op=ast.Add(), right=ast.Constant(value=r))

def macro_chain(*elts: ast.AST) -> ast.AST:
  """Build a tuple AST from any number of AST arguments."""
  return ast.Tuple(elts=list(elts), ctx=ast.Load())

# ---------------------------------------------------------------------------
# MacroExpander pass
# ---------------------------------------------------------------------------
class MacroExpander(SimplifyPass):
  """Inline calls whose parameters are annotated with ``: AST``.

  * For each parameter annotated with ``ast.AST`` we pass the original
    argument *AST* (un-evaluated).
  * For all other parameters we evaluate the argument expression in the
    current environment.
  * If the invoked function returns an ``ast.AST`` instance, we splice
    that AST back into the tree; otherwise we leave the call intact.
  """

  TESTS = [
    ("macro_add(x, 3)",  "x + 3"),
    ("macro_chain(a, b, c)",  "(a, b, c)"),
  ]
  TEST_ENV = { 'macro_add': macro_add, 'macro_chain': macro_chain }

  def visit_Call(self, node: ast.Call) -> ast.AST:
    self.generic_visit(node)

    try:
      func_obj = _eval_ast(node.func, self.env)
    except Exception:
      return node

    sig = inspect.signature(func_obj)
    hints = {}
    try:
      hints = get_type_hints(func_obj)
    except Exception:
      pass

    params = list(sig.parameters.values())

    # Build positional argument list, sending raw AST for AST-annotated params
    pos_args: List[Any] = []
    arg_iter = iter(node.args)
    for p in params:
      ann = hints.get(p.name)
      if p.kind in (inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD):
        try:
          arg_ast = next(arg_iter)
        except StopIteration:
          break
        if ann is ast.AST:
          pos_args.append(arg_ast)
        else:
          pos_args.append(_eval_ast(arg_ast, self.env))
      elif p.kind is inspect.Parameter.VAR_POSITIONAL:
        rest = list(arg_iter)
        if ann is ast.AST:
          pos_args.extend(rest)
        else:
          pos_args.extend(_eval_ast(a, self.env) for a in rest)
        break

    # Build keyword arguments
    kw_args: Dict[str, Any] = {}
    for kw in node.keywords:
      if kw.arg is None:
        return node
      param = sig.parameters.get(kw.arg)
      ann = hints.get(kw.arg)
      if param and ann is ast.AST:
        kw_args[kw.arg] = kw.value
      else:
        kw_args[kw.arg] = _eval_ast(kw.value, self.env)

    try:
      result = func_obj(*pos_args, **kw_args)
    except Exception:
      return node

    if isinstance(result, ast.AST):
      return result
    return node
