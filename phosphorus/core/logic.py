import itertools
import string
import inspect
from collections.abc import Iterable

"""phosphorus.core.logic
Light‑weight predicate‑logic helpers for an intro semantics course.

"""

# ╭─ 1 · Domain & handy global constants ──────────────────────╮
DOMAIN: tuple[str] = tuple(string.ascii_uppercase)
A, B, C, D, E, *_ = DOMAIN
# ╰────────────────────────────────────────────────────────────╯

# ────────────────── Utility helpers ──────────────────────────

def _tuplify(x):
  """If *x* is a single atom, return ``(x,)``; if it is already a tuple,
  leave it unchanged.  This is *idempotent*: ``_tuplify(_tuplify(x))``
  is always a tuple.
  """
  return (x,) if not isinstance(x, tuple) else x


def _tuplify_set(s: Iterable):
  """Return a *set* whose elements are all tuples, by applying
  ``_tuplify`` to each element of *s*.
  """
  return {_tuplify(t) for t in s}

# ────────────────── Core Relation class ──────────────────────

class Relation:
  """Boolean‑valued *k*‑ary relation stored as a set of tuples."""

  __slots__ = ("_ext", "arity")

  # construction ─────────────────────────────────────────────
  def __init__(self, extension: Iterable):
    self._ext = _tuplify_set(extension)
    if not self._ext:
      raise ValueError("Extension cannot be empty – arity undefined.")
    sizes = {len(t) for t in self._ext}
    if len(sizes) != 1:
      raise ValueError(f"Mixed‑arity tuples: {sizes}")
    self.arity = sizes.pop()

  # call behaviour ------------------------------------------
  def __call__(self, *args):
    if len(args) != self.arity:
      raise TypeError(f"Expected {self.arity} args, got {len(args)}")
    return int(tuple(args) in self._ext)

  # dictionary‑like access ----------------------------------
  def __getitem__(self, key):
    """Fix the first *m* arguments of the relation.

    * ``key`` may be an individual (for unary → k‑ary with k>1) or a
      tuple prefix `(a₁,…,a_m)` with `m < arity`.
    * If **exactly one** tuple in the extension begins with that prefix,
      return its *suffix* — a single element for binary relations or a
      tuple otherwise.  Otherwise raise :class:`KeyError`.
    """
    prefix = _tuplify(key)
    if len(prefix) >= self.arity:
      raise KeyError("Prefix length must be < relation arity")
    matches = [t[len(prefix):] for t in self._ext if t[:len(prefix)] == prefix]
    if len(matches) != 1:
      raise KeyError(f"{len(matches)} matches for prefix {prefix}")
    suffix = matches[0]
    return suffix[0] if len(suffix) == 1 else suffix

  # iteration & size ----------------------------------------
  def __iter__(self):
    return iter(self._ext)

  def __len__(self):
    return len(self._ext)

  # internal helpers ----------------------------------------
  def _ext_of(self, other):
    return other._ext if isinstance(other, Relation) else Relation(other)._ext

  def _new(self, ext):
    return Relation(ext)

  # native set algebra --------------------------------------
  def __or__(self, other):
    return self._new(self._ext | self._ext_of(other))

  def __and__(self, other):
    return self._new(self._ext & self._ext_of(other))

  def __sub__(self, other):
    return self._new(self._ext - self._ext_of(other))

  def __xor__(self, other):
    return self._new(self._ext ^ self._ext_of(other))

  # reflected operators -------------------------------------
  __ror__  = __or__
  __rand__ = __and__
  def __rsub__(self, other):
    return Relation(self._ext_of(other) - self._ext)
  __rxor__ = __xor__

  # pretty print --------------------------------------------
  def __repr__(self):
    elems = ", ".join(map(str, sorted(self._ext)))
    return f"Relation({{{elems}}})"

  # charset --------------------------------------------------
  def charset(self):
    if not all(isinstance(t[-1], int) and t[-1] in (0, 1) for t in self._ext):
      raise ValueError("charset only defined when last element is 0/1")
    prefixes = {t[:-1] for t in self._ext if t[-1] == 1}
    return {p[0] if len(p) == 1 else p for p in prefixes}

# alias for convenience in worksheets
Predicate = Relation

# ────────────────── Public helpers ───────────────────────────

def charset(rel: Relation):
  """Module‑level wrapper around :pymeth:`Relation.charset`."""
  return rel.charset()


def charfunc(s: Iterable):
  """Return the characteristic‑function relation for set *s*.

  ``s`` may contain individuals or tuples (all of the same arity).  The
  result is a k‑place relation where the *last* component is a boolean
  (1 for members of *s*, 0 otherwise).
  """
  core = _tuplify_set(s)
  if not core:
    raise ValueError("charfunc needs a non‑empty set to infer arity")
  arity = len(next(iter(core)))
  full = set()
  for tup in itertools.product(DOMAIN, repeat=arity):
    full.add(tup + (1 if tup in core else 0,))
  return Relation(full)


def single(obj):
  return bool(obj) and len(obj) == 1


def empty(rel):
  return len(rel) == 0


def nonempty(rel):
  return not empty(rel)

# ────────────────── Model container ──────────────────────────

class Model:
  def __init__(self, DOMAIN: Iterable[str] = DOMAIN):
    self.DOMAIN = tuple(DOMAIN)

  # individuals ---------------------------------------------
  def consts(self, names: str | Iterable[str]):
    if isinstance(names, str):
      names = names.split()
    for n in names:
      setattr(self, n, n)
    return self

  # relations ------------------------------------------------
  def pred(self, name: str, extension: Iterable):
    setattr(self, name, Relation(extension))
    return self

  def func(self, name: str, true_set: Iterable):
    setattr(self, name, charfunc(true_set))
    return self

  # convenience ---------------------------------------------
  def expose(self, g=None):
    if g is None:
      g = inspect.currentframe().f_back.f_globals  # caller's globals
    for k, v in self.__dict__.items():
      if k.isupper():
        g[k] = v
    return self


def expose(model: 'Model', g=None):
  return model.expose(g)

# ────────────────── exports ─────────────────────────────────
__all__ = [
  "DOMAIN", "A", "B", "C", "D", "E",
  "Relation", "Predicate",
  "charset", "charfunc", "single", "empty", "nonempty",
  "Model", "expose",
]

# ────────────────── quick self‑test ─────────────────────────
if __name__ == "__main__":
  # characteristic function of {A,C}
  f = charfunc({"A", "C"})
  assert f["A"] == 1 and f["B"] == 0
  assert charset(f) == {"A", "C"}

  # unary predicate & set algebra
  S = Relation({"A", "B"})
  assert S("A") == 1
  assert single({"A"})
  try:
    (({"C"} | S) - {("A",)}).charset()
  except ValueError:
    pass
  else:
    raise AssertionError("charset should have raised ValueError")

  print("All quick checks passed.")
