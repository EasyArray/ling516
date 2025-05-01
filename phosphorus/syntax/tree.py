'''
phosphorus.syntax.tree

Light wrapper around nltk.Tree that adds:
- .sem slot for semantic value
- nice HTML and SVG representations with subscripts via svgling
'''
from __future__ import annotations

import svgling
from nltk import Tree as _NLTKTree
from phosphorus.core.phivalue import PhiValue


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
        t = super().fromstring(s, **kwargs)
        t.__class__ = cls
        t.sem = getattr(t, 'sem', None)
        return t

    def _xrepr_html_(self) -> str:
        """
        HTML representation for Jupyter/Colab: preformatted text.
        """
        return f"<pre>{self.pformat(margin=60)}</pre>"

    def _repr_svg_(self) -> str:
        """
        SVG representation for Jupyter: draws the tree with svgling,
        using split_leaf to render '_' as subscript.
        """
        return svgling.draw_tree(self, tree_split=split_leaf)._repr_svg_()
