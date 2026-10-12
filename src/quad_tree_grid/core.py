"""Core quadtree implementation.

A *region* quadtree: the domain is a fixed bounding box recursively split into
four equal children when a capacity threshold is exceeded. Points are stored at
leaf nodes only. This structure is good for uniformly-distributed 2D point data
where the primary access pattern is "give me everything inside this rectangle".

Design decisions
----------------
* Points are stored as ``(x, y)`` tuples, not a custom class, so callers can
  pass plain tuples without wrapping. ``Point`` is just an alias for typing.
* Bounding boxes use the half-open convention ``[x0, x1) x [y0, y1)``. This is
  chosen so that when a node splits, its four children tile the parent with
  no overlap and no gaps — the right edge of the left child is exactly the left
  edge of the right child, and a point on that boundary belongs to exactly one
  child. This eliminates ambiguous placement of boundary points.
* Splitting is depth-limited by an optional ``max_depth`` to prevent runaway
  recursion on degenerate input (e.g. many points with identical coordinates).
  When ``max_depth`` is reached the node becomes an unbounded leaf regardless
  of ``capacity``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

Point = Tuple[float, float]

__version__ = "1.0.0"


class BBox:
    """Axis-aligned bounding box with half-open intervals.

    A point ``(x, y)`` is contained iff ``x0 <= x < x1`` and ``y0 <= y < y1``.
    The half-open convention makes quadtree child regions tile perfectly.
    """

    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0: float, y0: float, x1: float, y1: float) -> None:
        if x1 <= x0:
            raise ValueError(f"x1 ({x1}) must be > x0 ({x0})")
        if y1 <= y0:
            raise ValueError(f"y1 ({y1}) must be > y0 ({y0})")
        self.x0 = float(x0)
        self.y0 = float(y0)
        self.x1 = float(x1)
        self.y1 = float(y1)

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def intersects(self, other: "BBox") -> bool:
        """Half-open intersection test.

        Two boxes intersect iff their closed interiors overlap. With half-open
        intervals, abutting boxes (e.g. ``[0,1)`` and ``[1,2)``) are NOT
        considered intersecting, which is correct for query semantics here: a
        range query over ``[1, 2)`` should visit the child whose region is
        ``[1, 2)`` but not its sibling ``[0, 1)``.
        """
        if other.x1 <= self.x0 or self.x1 <= other.x0:
            return False
        if other.y1 <= self.y0 or self.y1 <= other.y0:
            return False
        return True

    def quadrants(self) -> Tuple["BBox", "BBox", "BBox", "BBox"]:
        """Return the four equal child regions (nw, ne, sw, se).

        Because the boxes are half-open, the four children exactly partition the
        parent: every point in the parent belongs to exactly one child.
        """
        mx = (self.x0 + self.x1) / 2
        my = (self.y0 + self.y1) / 2
        return (
            BBox(self.x0, my, mx, self.y1),  # NW
            BBox(mx, my, self.x1, self.y1),  # NE
            BBox(self.x0, self.y0, mx, my),  # SW
            BBox(mx, self.y0, self.x1, my),  # SE
        )

    def __repr__(self) -> str:
        return f"BBox({self.x0}, {self.y0}, {self.x1}, {self.y1})"


class _Node:
    """Internal quadtree node.

    A node is either a leaf (``children is None``, points stored in ``points``)
    or an interior node (``points`` empty, ``children`` holds four nodes).
    """

    __slots__ = ("bbox", "capacity", "max_depth", "depth", "points", "children")

    def __init__(
        self, bbox: BBox, capacity: int, max_depth: int, depth: int
    ) -> None:
        self.bbox = bbox
        self.capacity = capacity
        self.max_depth = max_depth
        self.depth = depth
        self.points: List[Point] = []
        self.children: Optional[List[_Node]] = None

    def _is_full_leaf(self) -> bool:
        return self.children is None and len(self.points) >= self.capacity

    def _can_split(self) -> bool:
        # Only split if the current depth permits going one level deeper.
        return self.depth < self.max_depth

    def _insert(self, x: float, y: float) -> None:
        if not self.bbox.contains(x, y):
            return

        if self.children is not None:
            for c in self.children:
                c._insert(x, y)
            return

        self.points.append((x, y))

        if self._is_full_leaf() and self._can_split():
            self._split()

    def _split(self) -> None:
        """Subdivide this leaf into four children and redistribute points."""
        quads = self.bbox.quadrants()
        self.children = [
            _Node(q, self.capacity, self.max_depth, self.depth + 1) for q in quads
        ]
        old = self.points
        self.points = []
        for (px, py) in old:
            for c in self.children:
                if c.bbox.contains(px, py):
                    c.points.append((px, py))
                    break

    def _query(self, box: BBox, out: List[Point]) -> None:
        if not self.bbox.intersects(box):
            return
        if self.children is None:
            for (px, py) in self.points:
                if box.contains(px, py):
                    out.append((px, py))
        else:
            for c in self.children:
                c._query(box, out)


class QuadTree:
    """Region quadtree over a fixed bounding box.

    Args:
        x0, y0, x1, y1: bounds of the root region. A point is insertable iff
            it lies within this region (half-open).
        capacity: max points per leaf before the leaf splits.
        max_depth: hard depth limit. A leaf at this depth holds points without
            further splitting, which handles clustered/degenerate input.

    The tree raises ``ValueError`` for points outside the root bounds or for an
    invalid bounding box. A point lying exactly on the left/top boundary is in;
    a point on the right/bottom boundary is out (half-open convention).
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        capacity: int = 8,
        max_depth: int = 20,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        self.root = _Node(BBox(x0, y0, x1, y1), capacity, max_depth, depth=1)
        self.capacity = capacity
        self.max_depth = max_depth
        self._count = 0

    def insert(self, x: float, y: float) -> None:
        """Insert a point. Raises ``ValueError`` if outside root bounds."""
        if not self.root.bbox.contains(x, y):
            raise ValueError(
                f"point ({x}, {y}) outside tree bounds "
                f"[{self.root.bbox.x0}, {self.root.bbox.x1}) x "
                f"[{self.root.bbox.y0}, {self.root.bbox.y1})"
            )
        self.root._insert(x, y)
        self._count += 1

    def insert_all(self, points: Sequence[Point]) -> int:
        """Insert many points. Returns count successfully inserted.

        Insertion stops at the first point that raises (so earlier points are
        already committed). The return value is the number inserted before any
        error.
        """
        n = 0
        for (x, y) in points:
            self.insert(x, y)
            n += 1
        return n

    def query(self, x0: float, y0: float, x1: float, y1: float) -> List[Point]:
        """Return all points within ``[x0, x1) x [y0, y1)``.

        The query box is interpreted with the same half-open convention as the
        tree. If ``x1 <= x0`` or ``y1 <= y0`` the result is empty.
        """
        if x1 <= x0 or y1 <= y0:
            return []
        out: List[Point] = []
        self.root._query(BBox(x0, y0, x1, y1), out)
        return out

    def __len__(self) -> int:
        return self._count

    def count(self) -> int:
        """Number of points stored (same as ``len(tree)``)."""
        return self._count
