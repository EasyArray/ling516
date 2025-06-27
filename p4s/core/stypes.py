"""
phosphorus.core.stypes
---------------------

Minimal Heim‑&‑Kratzer semantic types with a *very* compact DSL:

  >>> Type.e               # atomic
  >>> Type.et              # e → t
  >>> Type.eet             # e → (e → t)
  >>> Type.eet__et         # ((e→(e→t)) → (e→t)) – underscores left‑associate

Run this file (``python types.py``) to execute a basic self‑test suite.
"""

from __future__ import annotations

from functools import reduce
from itertools import count
from typing import Any, Dict, Tuple

# ---------------------------------------------------------------------------
#  Type metaclass
# ---------------------------------------------------------------------------

class _TypeMeta(type):
  """Create and *intern* :class:`Type` objects from attribute suffixes."""

  _suffix_cache: Dict[str, "Type"] = {}
  _fresh_counter = count()

  # -------------------------------------------------------------------
  #  public helpers
  # -------------------------------------------------------------------

  def __getattr__(cls, suffix: str) -> "Type":
    """Allow ``Type.et``‑style access; result is memoised per suffix."""
    if suffix not in cls._suffix_cache:
      cls._suffix_cache[suffix] = cls._build_from_suffix(suffix)
    return cls._suffix_cache[suffix]

  def fresh(cls, label: str | None = None) -> "Type":
    n = next(cls._fresh_counter)
    return cls((f"?{label or 'τ'}{n}",))

  # -------------------------------------------------------------------
  #  internal parsing
  # -------------------------------------------------------------------

  @staticmethod
  def _build_from_suffix(s: str) -> "Type":
    if not s or any(ch not in "et_" for ch in s):
      raise AttributeError(f"invalid type suffix '{s}'")

    # 1. build stack, folding '_' on the fly
    stack: list[Type] = []
    for ch in s:
      if ch == "_":
        if len(stack) < 2:
          raise AttributeError("misplaced '_' in type suffix")
        right = stack.pop()
        left = stack.pop()
        stack.append(Type((left, right)))
      else:
        stack.append(Type((ch,)))  # atomic; constructor handles interning

    # 2. fold remaining stack right‑associatively
    def fold(acc: Type, nxt: Type) -> Type:  # type: ignore[valid-type]
      return Type((nxt, acc))

    return reduce(fold, reversed(stack))

# ---------------------------------------------------------------------------
#  Type class with *constructor‑level* interning
# ---------------------------------------------------------------------------

class Type(tuple, metaclass=_TypeMeta):
  """Immutable semantic type (atomic or ⟨domain, range⟩)."""

  __slots__: Tuple[str, ...] = ()
  _intern_cache: Dict[Tuple[Any, ...], "Type"] = {}

  # ---------------------------------------------------------------
  #  constructor
  # ---------------------------------------------------------------

  def __new__(cls, items):  # type: ignore[override]
    tpl = items if isinstance(items, tuple) else (items,)
    if tpl in cls._intern_cache:
      return cls._intern_cache[tpl]
    obj = tuple.__new__(cls, tpl)
    cls._intern_cache[tpl] = obj
    return obj

  @classmethod
  def from_spec(cls, spec: str | tuple) -> "Type":  # type: ignore[valid-type]
    """Build a semantic Type from a string or nested tuple spec.

    Examples:
      'et'         -> Type.et
      ('e','t')    -> Type((Type.e,Type.t))
      ('et','et_t')-> Type((Type.et, Type.et_t))
    """
    match spec:
      case str() as s:
        # simple suffix
        return cls._build_from_suffix(s)

      case tuple() as tup:
        # nested tuple: build each element then fold right-associatively
        parts = [cls.from_spec(elem) for elem in tup]

        # fold remaining stack right-associatively
        def fold(acc: "Type", nxt: "Type") -> "Type":  # type: ignore[valid-type]
          return cls((nxt, acc))
        return reduce(fold, reversed(parts))

      case _:
        raise TypeError(f"Invalid spec type: {spec!r}")

  # predicates ----------------------------------------------------

  @property
  def is_atomic(self) -> bool:
    return len(self) == 1

  @property
  def is_function(self) -> bool:
    return len(self) == 2
  
  @property
  def is_unknown(self) -> bool:
    return len(self) == 1 and self[0].startswith("?")

  # components ----------------------------------------------------

  @property
  def domain(self) -> "Type":
    if not self.is_function:
      raise ValueError("atomic type has no domain")
    return self[0]

  @property
  def range(self) -> "Type":
    if not self.is_function:
      raise ValueError("atomic type has no range")
    return self[1]

  # representation ------------------------------------------------

  def __repr__(self) -> str:  # noqa: D401
    if self.is_atomic:
      return self[0]
    return f"({self.domain}→{self.range})"


# ---------------------------------------------------------------------------
#  Convenience aliases   (avoid clash with capital‑letter individuals)
# ---------------------------------------------------------------------------

e = Type.e  # noqa: E741
t = Type.t
et = Type.et


# ---------------------------------------------------------------------------
#  Tiny helper
# ---------------------------------------------------------------------------

def takes(func: Any, arg: Any) -> bool:
  """Return ``True`` iff *func* has type ⟨A,B⟩ and *arg* has type A."""
  ft = getattr(func, "stype", None)
  at = getattr(arg, "stype", None)
  return bool(ft and ft.is_function and ft.domain == at)


# ---------------------------------------------------------------------------
#  Tests (run ``python types.py``)
# ---------------------------------------------------------------------------

def _run_tests() -> None:
  # atomic vs function
  assert Type.e.is_atomic and not Type.e.is_function
  assert Type.et.is_function and not Type.et.is_atomic

  # domain / range identity
  assert Type.et.domain is Type.e
  assert Type.et.range is Type.t

  # constructor‑level caching
  assert Type(("e",)) is Type.e
  assert Type((Type.e, Type.t)) is Type.et

  # dsl parsing
  assert repr(Type.eet) == "(e→(e→t))"
  assert repr(Type.eet__et) == "((e→(e→t))→(e→t))"

  # deeper structure
  assert Type.eet.domain is Type.e
  assert Type.eet.range.domain is Type.e
  assert Type.eet.range.range is Type.t

  # nested underscores
  assert repr(Type.eet__et__et) == "(((e→(e→t))→(e→t))→(e→t))"

  # -------------------------------------------------------------------
  # Tests for from_spec
  # -------------------------------------------------------------------

  # Nested function type
  assert Type.from_spec("eet") is Type.eet
  assert repr(Type.from_spec("eet")) == "(e→(e→t))"

  # Tuple-based specification
  assert Type.from_spec(("e", "t")) is Type.et
  assert repr(Type.from_spec(("e", "et"))) == "(e→(e→t))"

  # Deeply nested tuple specification
  assert repr(Type.from_spec((("e", "t"), "et"))) == "((e→t)→(e→t))"

  # Invalid specifications
  try:
    Type.from_spec("invalid")
  except AttributeError as er:
    assert str(er) == "invalid type suffix 'invalid'"

  try:
    Type.from_spec(("e", "invalid"))
  except AttributeError as er:
    assert str(er) == "invalid type suffix 'invalid'"

  try:
    Type.from_spec(123)
  except TypeError as er:
    assert str(er) == "Invalid spec type: 123"

  print("✅  All type‑system tests passed, including from_spec tests.")


if __name__ == "__main__":
  _run_tests()
