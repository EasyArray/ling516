"""phosphorus.core.phivalue
---------------------------------
Light‑weight wrapper around a Python AST that stores an optional
semantic type and guard/presupposition and beta‑reduces the AST
at construction time.

All heavy work (pretty HTML, richer type inference) lives elsewhere.
"""

import ast
import copy
from collections import ChainMap
from typing import Any, Optional

from p4s.simplify           import simplify          # local functional API
from p4s.simplify.utils     import capture_env       # caller env snapshot
from p4s.core.display       import render_phi_html   # rich HTML helper
from p4s.core.infer         import infer_and_strip   # type checker / DSL stripper
from p4s.core.stypes        import Type              # semantic type system
from p4s.core.constants     import UNDEF             # sentinel for undefined values


class _EvaluatedLambda:
  """Callable wrapper with a stable semantic repr for evaluated lambdas."""

  __slots__ = ("_fn", "_preview")

  def __init__(self, fn, preview: str):
    self._fn = fn
    self._preview = preview

  def __call__(self, *args, **kwargs):
    return self._fn(*args, **kwargs)

  def __repr__(self):
    return self._preview

  def __str__(self):
    return self._preview


def _lambda_param_names(expr: ast.Lambda) -> set[str]:
  names = {
    a.arg for a in (
      list(expr.args.posonlyargs)
      + list(expr.args.args)
      + list(expr.args.kwonlyargs)
    )
  }
  if expr.args.vararg is not None:
    names.add(expr.args.vararg.arg)
  if expr.args.kwarg is not None:
    names.add(expr.args.kwarg.arg)
  return names


def _lambda_header(expr: ast.Lambda) -> str:
  preview_lambda = ast.Lambda(args=copy.deepcopy(expr.args), body=ast.Constant(value=None))
  return ast.unparse(preview_lambda).rsplit(":", 1)[0]


class _GuardModToIfExp(ast.NodeTransformer):
  def visit_BinOp(self, node: ast.BinOp):
    left_was_guard = isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mod)
    node = self.generic_visit(node)
    if isinstance(node.op, ast.Mod):
      if not left_was_guard:
        return ast.IfExp(
          test=node.right,
          body=node.left,
          orelse=ast.Name(id=str(UNDEF), ctx=ast.Load()),
        )

      lhs_name = "__guard_lhs"
      lhs_ref = ast.Name(id=lhs_name, ctx=ast.Load())
      undef_ref = ast.Name(id=str(UNDEF), ctx=ast.Load())
      return ast.Call(
        func=ast.Lambda(
          args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg=lhs_name)],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            vararg=None,
            kwarg=None,
          ),
          body=ast.IfExp(
            test=ast.Compare(
              left=lhs_ref,
              ops=[ast.Is()],
              comparators=[undef_ref],
            ),
            body=ast.Name(id=str(UNDEF), ctx=ast.Load()),
            orelse=ast.IfExp(
              test=node.right,
              body=lhs_ref,
              orelse=ast.Name(id=str(UNDEF), ctx=ast.Load()),
            ),
          ),
        ),
        args=[node.left],
        keywords=[],
      )
    return node


def _eval_ast_with_guards(expr: ast.AST, env: dict[str, object]) -> Any:
  # Keep runtime guard semantics here. simplify/guard_pass.py handles
  # static normalization, while lambda preview reuses this evaluator so
  # display stays aligned with actual PhiValue execution.
  expr = copy.deepcopy(expr)
  if any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
         for n in ast.walk(expr)):
    expr = _GuardModToIfExp().visit(expr)
    ast.fix_missing_locations(expr)
  code = compile(ast.Expression(expr), filename="<phivalue>", mode="eval")
  return eval(code, env)


def _lambda_has_global_false_guard(expr: ast.Lambda, env: dict[str, object]) -> bool:
  """True when a lambda body has a `% guard` independent of params that is false."""
  params = _lambda_param_names(expr)

  node = expr.body
  guards: list[ast.AST] = []
  while isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
    guards.append(node.right)
    node = node.left

  for guard in guards:
    guard_uses_param = any(
      isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in params
      for n in ast.walk(guard)
    )
    if guard_uses_param:
      continue
    try:
      value = _eval_ast_with_guards(guard, env)
    except Exception:
      continue
    if not bool(value):
      return True

  return False


def _lambda_preview(expr: ast.Lambda, env: dict[str, object]) -> str:
  """Compact textual form for evaluated lambda callables."""
  header = _lambda_header(expr)
  if _lambda_has_global_false_guard(expr, env):
    return f"{header}: UNDEF"

  params = _lambda_param_names(expr)
  uses_param = any(
    isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in params
    for n in ast.walk(expr.body)
  )
  if uses_param:
    return ast.unparse(expr)
  try:
    value = _eval_ast_with_guards(expr.body, env)
  except Exception:
    return f"{header}: {ast.unparse(expr.body)}"

  if value is UNDEF:
    rendered = "UNDEF"
  else:
    rendered = repr(value)
  return f"{header}: {rendered}"

# ---------------------------------------------------------------------------
#  PhiValue
# ---------------------------------------------------------------------------

class PhiValue:
  """An AST + optional semantic type and guard with Jupyter‑friendly HTML."""

  __slots__ = ("expr", "stype", "guard", "_env")

  # ---------------------------------------------------------------------
  #  construction
  # ---------------------------------------------------------------------

  def __init__(self,
               expr: ast.AST | str, *,
               stype: Optional[Type] = None,
               guard: Optional[ast.AST] = None) -> None:
    # 0: parse or accept AST
    if isinstance(expr, str):
      expr = ast.parse(expr, mode="eval").body
    elif isinstance(expr, (int, float, bool, str)):
      # Convert basic Python literals to AST constants
      expr = ast.Constant(value=expr)

    # 1. capture *caller* environment (skip this frame)
    env = capture_env(skip=1)          # ChainMap

    # 2. *Infer* type while DSL cues are still present (also strips DSL cues)
    expr = infer_and_strip(expr, env)
    inferred_type = getattr(expr, "stype", None)
    inferred_guard = getattr(expr, "guard", None)

    # 3. beta‑reduce / macro‑expand (pure)
    simplified = simplify(expr, env=env)
    simplified_guard = inferred_guard and simplify(guard or inferred_guard, env=env)

    # 4. store
    self.expr  = simplified
    self._env  = ChainMap({}, env)     # make a shallow, isolated view
    self.stype = stype or inferred_type or getattr(simplified, "stype", None)
    self.guard = simplified_guard

  # ---------------------------------------------------------------------
  #  functional behaviour
  # ---------------------------------------------------------------------

  def __call__(self, *args: "PhiValue", **kwargs) -> Any:
    # Convert basic types to PhiValues
    args = tuple(PhiValue(a) if not isinstance(a, PhiValue) else a for a in args)
    kwargs = {k: PhiValue(v) if not isinstance(v, PhiValue) else v for k, v in kwargs.items()}
    
    # Attach type info BEFORE deepcopy so it gets copied
    if self.stype is not None:
      self.expr.stype = self.stype
    for a in args:
      if a.stype is not None:
        a.expr.stype = a.stype
    
    call_ast = ast.Call(
      func=copy.deepcopy(self.expr),
      args=[copy.deepcopy(a.expr) for a in args],
      keywords=[ast.keyword(arg=k, value=copy.deepcopy(kwargs[k].expr)) for k in kwargs]
    )
    phi = PhiValue(call_ast)
    try:
      result = phi.eval()
      # Keep Python callables (e.g., lambda/function objects) as unevaluated
      # PhiValue for readable display, but return evaluated PhiValue results.
      if callable(result) and not isinstance(result, PhiValue):
        return phi
      return result
    except Exception:
      return phi

  # ---------------------------------------------------------------------
  #  evaluation helpers
  # ---------------------------------------------------------------------

  def eval(self) -> Any:
    """Evaluate the stored expression in its captured environment."""
    # Python's eval requires a real dict for globals
    env_dict = dict(self._env)
    out = _eval_ast_with_guards(self.expr, env_dict)
    if out is UNDEF:
      return UNDEF
    if self.stype == Type.t:
      out = int(bool(out))
    if callable(out) and isinstance(self.expr, ast.Lambda):
      return _EvaluatedLambda(out, _lambda_preview(self.expr, env_dict))
    return out

  # ---------------------------------------------------------------------
  #  dunder utilities
  # ---------------------------------------------------------------------

  def __bool__(self):
    return bool(self.eval())

  def __hash__(self):
    guard_dump = None if self.guard is None else ast.dump(self.guard, annotate_fields=False)
    return hash((
      ast.dump(self.expr, annotate_fields=False),
      self.stype,
      guard_dump,
    ))

  def __eq__(self, other):
    if isinstance(other, PhiValue):
      if (ast.dump(self.expr, annotate_fields=False) !=
          ast.dump(other.expr, annotate_fields=False)):
        return False

      if self.stype != other.stype:
        return False

      if self.guard == other.guard == None:
        return True

      if self.guard is None or other.guard is None:
        return False

      return (ast.dump(self.guard, annotate_fields=False) ==
                 ast.dump(other.guard, annotate_fields=False))
    
    try:
      return self.eval() == other
    except:
      return NotImplemented

  def __mod__(self, guard):
    if not guard:
      return UNDEF
    return self

  def __repr__(self):
    return ast.unparse(self.expr)

  # Jupyter rich repr
  def _repr_html_(self):
    #return f"<code>{ast.unparse(self.expr)}</code>  <small>{self.stype}</small>"
    return render_phi_html(self.expr, stype=self.stype)

# ---------------------------------------------------------------------------
#  rudimentary tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
  id_ast = ast.parse("lambda x=t: x[t]", mode="eval").body
  id_pv = PhiValue(id_ast)
  two_pv = PhiValue(ast.parse("2", mode="eval").body)

  assert id_pv(two_pv).eval() == 2
  assert id_pv == id_pv
  assert isinstance(hash(id_pv), int)
  print("✅ PhiValue sanity tests passed.")
