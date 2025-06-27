'''
phosphorus.syntax.tree

Light wrapper around nltk.Tree that adds:
- .sem slot for semantic value
- nice HTML and SVG representations with subscripts via svgling
'''
from __future__ import annotations

import ast
import html
import svgling
from nltk import Tree as _NLTKTree
from p4s.core.phivalue import PhiValue
from p4s.core.display import make_badge_html, make_code_html, _CSS

import svgling.html as svh
from svgling.core import VertAlign
from xml.etree.ElementTree import Element, SubElement
import xml.etree.ElementTree as ET
from black import format_str, Mode

def _sem_label(label_str: str, sem_obj) -> Element:
  """
  Return an Element that shows `label_str` on the first line and
  `sem_obj` (rendered via its _repr_html_) on the second line.
  """
  outer = Element(                      # grid holds the two lines
      "div",
      style="display:inline-grid;grid-template-columns:auto"
  )

  # 1st line – the syntactic category
  l1 = SubElement(
      outer, "div",
      style="padding:0 .75em;text-align:center;"
  )
  l1.append(svh.to_html(label_str))      # plain text wrapper

  # 2nd line – full HTML from the semantic value
  l2 = SubElement(
      outer, "div",
      style="padding:0 .75em;text-align:center;"
  )
  l2.append(svh.to_html(sem_obj))        # <-- critical: give the object itself

  return outer

def xsplit_with_sem(node):
  label, kids = split_leaf(node)
  sem = getattr(node, "sem", None)
  if sem is not None:
    return _sem_label(label, sem), kids
  return label, kids

def _label_to_html(label: str) -> str:
  if '_' in label:
      base, sub = label.split('_', 1)
      label = f"{html.escape(base)}<sub>{html.escape(sub)}</sub>"
  else:
    label = html.escape(label)
  return f"<span>{label}</span>"

def split_with_sem(node):
  """
  Tree split function that embeds a syntactic label with an inline badge
  and places the pretty-printed, highlighted code block below.

  Returns:
      label_element (ET.Element): XML element for svgling to embed directly.
      children (tuple): child nodes of the tree node.
  """
  # 1) Extract label and children
  if isinstance(node, Tree):
    label, kids = node.label(), tuple(node)
  else:
    label, kids = node, ()
  label = _label_to_html(label)  # escape label for HTML

  sem = getattr(node, 'sem', None)
  if sem is None:
    return ET.fromstring(label), kids
  

  # 2) Build inline badge next to the syntactic label
  typ = getattr(sem, 'stype', None)
  badge_html = make_badge_html(typ, font_size='11px') if typ else ''
  first_line = f"<span>{label} {badge_html}</span>"

  # 3) Pretty-print code with Black at a narrower line length for trees
  expr = getattr(sem, 'expr', None)
  if expr:
    pretty = format_str(ast.unparse(sem.expr), mode=Mode(line_length=50))
    code_html   = make_code_html(pretty, font_size='11px')
    max_chars = max(len(line) for line in pretty.splitlines())
    width_css = f"min-width:{max_chars + 2}ch;"   # +x for padding
  else:
    code_html = sem._repr_html_() if hasattr(sem, '_repr_html_') else str(sem)
    width_css = ''
    
  # 4) Wrap both lines in a single <div> grid for vertical stacking
  combined_html = (
    "<div style='display:inline-grid;"
    "grid-auto-flow:row;justify-items:center;padding:0 .75em;"
    f"{width_css}'>"
    f'{_CSS}'
    f"{first_line}"
    f"<div style='text-align:left;'>{code_html}</div></div>"
  )

  # 5) Parse the HTML into an Element, ensuring a single root for svgling
  combined_el = ET.fromstring(combined_html)

  return combined_el, kids

def split_leaf(node):
  """
  Helper for svgling: split a node label on '_' into a base and subscript.
  If the label contains exactly one underscore, renders the part after
  the underscore as a subscript via svgling.core.subscript_node.
  """
  if isinstance(node, str):
    label, children = node, ()
  else:
    label = node.label()
    children = tuple(node)
  # split only on the first underscore to allow labels like 'X_Y_Z'
  parts = label.split('_', 1)
  if len(parts) == 2:
    return svgling.core.subscript_node(parts[0], parts[1]), children
  return label, children

class Leaf(str):
  """
  A simple subclass of str that can be used as a leaf node in an nltk.Tree.
  Attributes:
      sem (PhiValue | None): semantic value attached by interpretation.
  """
  @property
  def sem(self):
    return self._sem

  @sem.setter
  def sem(self, value):
    self._sem = value

class Tree(_NLTKTree):
  """
  A syntax node that can cache a semantic value and render nicely.

  Attributes:
      sem (PhiValue | None): semantic value attached by interpretation.
  """

  __slots__ = ("sem",)
  __match_args__ = ("_label", "children")
  @property
  def children(self): return list(self)

  def __init__(self, label, children=(), *, sem: PhiValue | None = None):
    # Initialize as an nltk.Tree with given label and children
    super().__init__(label, children)
    # Semantic value (to be filled by Interpreter)
    self.sem = sem

  @classmethod
  def fromstring(cls, s: str, **kwargs) -> Tree:
    """
    Parse a bracketed string into a Tree (subclass).
    Ensures the resulting tree is our Tree class.
    """
    def leaf_to_tree(label):
      return Leaf(label)
    t = super().fromstring(s, read_leaf=leaf_to_tree, **kwargs)
    t.__class__ = cls
    t.sem = getattr(t, 'sem', None)
    return t

  def _xrepr_html_(self) -> str:
    """
    HTML representation for Jupyter/Colab: preformatted text.
    """
    return f"<pre>{self.pformat(margin=60)}</pre>"

  def x_repr_svg_(self) -> str:
    """
    SVG representation for Jupyter: draws the tree with svgling,
    using split_leaf to render '_' as subscript.
    """
    return svgling.draw_tree(self, tree_split=split_leaf)._repr_svg_()

  def _repr_html_(self) -> str:
    """
    HTML representation for Jupyter: draws the tree with svgling.html,
    stacking any node.sem HTML underneath its label and returns
    a <div>…</div> that Jupyter will render as HTML.
    """
  
    return svh.draw_tree(
      self,
      tree_split=split_with_sem,
      vert_align=VertAlign.CENTER,
      #average_glyph_width=0.6,     # default is 2.0  →  25 % wider slots
    )._to_html()
  