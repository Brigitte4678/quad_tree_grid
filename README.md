# quad_tree_grid

Stores 2D point data in a region quadtree for bounding-box range queries.

```python
from quad_tree_grid import QuadTree

t = QuadTree(0, 0, 100, 100, capacity=8, max_depth=20)
t.insert(10, 10)
t.insert(80, 80)
hits = t.query(0, 0, 50, 50)   # -> [(10, 10)]
```

`BBox` and `Point` are also exported; `Point` is a typing alias for `Tuple[float, float]`, so plain tuples are accepted everywhere.

## Why

For uniformly-distributed point data where the main access pattern is "give me every point inside this rectangle", a region quadtree gives O(depth) pruning per query. The trade-off versus a flat list scanned linearly is worse on small datasets (the tree pays overhead for splits and recursion) and better once the point count grows into the thousands and queries are selective.

## Edge cases

Bounding boxes are half-open: `[x0, x1) x [y0, y1)`. A point on the left or top edge of the root is inside; a point on the right or bottom edge is outside and will raise `ValueError` on insert. The same convention applies to `query`. This was chosen so that when a leaf splits, its four children tile the parent with no overlap and no gaps, and every point lands in exactly one child.

Degenerate input is handled by `max_depth`: when a leaf reaches `max_depth` it stops splitting and accumulates points without bound. The default is 20, which bounds recursion even for many identical points.
