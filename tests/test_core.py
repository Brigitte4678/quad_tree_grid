"""Unit tests for quad_tree_grid.core.

PYTHONPATH=src python -m unittest discover -s tests
"""

import unittest

from quad_tree_grid import BBox, Point, QuadTree


class TestBBox(unittest.TestCase):

    def test_contains_half_open(self):
        b = BBox(0, 0, 10, 10)
        # left/top edges in
        self.assertTrue(b.contains(0, 0))
        # right/bottom edges out
        self.assertFalse(b.contains(10, 0))
        self.assertFalse(b.contains(0, 10))
        self.assertTrue(b.contains(9, 9))

    def test_intersects_abutting_not_intersecting(self):
        # Half-open abutting boxes do not intersect.
        a = BBox(0, 0, 5, 5)
        b = BBox(5, 0, 10, 5)
        self.assertFalse(a.intersects(b))
        self.assertFalse(b.intersects(a))

    def test_intersects_overlap(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(4, 0, 10, 5)
        self.assertTrue(a.intersects(b))
        self.assertTrue(b.intersects(a))

    def test_intersects_disjoint(self):
        a = BBox(0, 0, 5, 5)
        b = BBox(6, 0, 10, 5)
        self.assertFalse(a.intersects(b))

    def test_quadrants_partition_origin(self):
        b = BBox(0, 0, 10, 10)
        nw, ne, sw, se = b.quadrants()
        # The shared midpoint belongs to exactly one child per axis.
        # (5,5) should be in NE: [5,10) x [5,10)
        self.assertFalse(nw.contains(5, 5))
        self.assertTrue(ne.contains(5, 5))
        self.assertFalse(sw.contains(5, 5))
        self.assertFalse(se.contains(5, 5))

    def test_invalid_bounds_raise(self):
        with self.assertRaises(ValueError):
            BBox(5, 0, 5, 10)  # x1 == x0
        with self.assertRaises(ValueError):
            BBox(0, 5, 10, 5)  # y1 == y0
        with self.assertRaises(ValueError):
            BBox(5, 0, 4, 10)  # x1 < x0


class TestQuadTreeInsert(unittest.TestCase):

    def test_insert_and_count(self):
        t = QuadTree(0, 0, 100, 100, capacity=4)
        t.insert(1, 1)
        t.insert(2, 2)
        self.assertEqual(len(t), 2)
        self.assertEqual(t.count(), 2)

    def test_insert_outside_bounds_raises(self):
        t = QuadTree(0, 0, 10, 10)
        with self.assertRaises(ValueError):
            t.insert(10, 5)  # right edge out
        with self.assertRaises(ValueError):
            t.insert(5, 10)  # bottom edge out
        with self.assertRaises(ValueError):
            t.insert(-1, 5)

    def test_insert_all_returns_count(self):
        t = QuadTree(0, 0, 100, 100, capacity=4)
        n = t.insert_all([(1, 1), (2, 2), (3, 3)])
        self.assertEqual(n, 3)
        self.assertEqual(len(t), 3)

    def test_insert_all_stops_on_error(self):
        t = QuadTree(0, 0, 10, 10)
        # (15,15) is out of bounds; first two should be committed.
        with self.assertRaises(ValueError):
            t.insert_all([(1, 1), (2, 2), (15, 15)])
        self.assertEqual(len(t), 2)

    def test_split_on_capacity(self):
        t = QuadTree(0, 0, 100, 100, capacity=2, max_depth=5)
        t.insert(1, 1)
        t.insert(2, 2)
        t.insert(3, 3)
        # third insert triggers split; root now has four children
        self.assertIsNotNone(t.root.children)
        self.assertEqual(len(t.root.children), 4)
        self.assertEqual(len(t.root.points), 0)
        self.assertEqual(len(t), 3)

    def test_max_depth_prevents_split(self):
        t = QuadTree(0, 0, 100, 100, capacity=2, max_depth=1)
        # depth 1 == root; cannot split deeper, so leaf holds all points.
        for i in range(10):
            t.insert(float(i), float(i))
        self.assertIsNone(t.root.children)
        self.assertEqual(len(t.root.points), 10)
        self.assertEqual(len(t), 10)

    def test_identical_points(self):
        t = QuadTree(0, 0, 100, 100, capacity=2, max_depth=6)
        for _ in range(5):
            t.insert(5, 5)
        # All identical: split until max_depth, then accumulate in one leaf.
        self.assertEqual(len(t), 5)
        results = t.query(0, 0, 100, 100)
        self.assertEqual(len(results), 5)
        for p in results:
            self.assertEqual(p, (5, 5))


class TestQuadTreeQuery(unittest.TestCase):

    def test_query_empty_tree(self):
        t = QuadTree(0, 0, 100, 100)
        self.assertEqual(t.query(0, 0, 100, 100), [])

    def test_query_all(self):
        t = QuadTree(0, 0, 100, 100, capacity=2)
        pts = [(10, 10), (20, 20), (80, 80), (90, 90)]
        for p in pts:
            t.insert(*p)
        got = t.query(0, 0, 100, 100)
        self.assertEqual(sorted(got), sorted(pts))

    def test_query_subregion(self):
        t = QuadTree(0, 0, 100, 100, capacity=2, max_depth=6)
        t.insert(5, 5)
        t.insert(15, 15)
        t.insert(95, 95)
        got = t.query(0, 0, 50, 50)
        self.assertIn((5, 5), got)
        self.assertIn((15, 15), got)
        self.assertNotIn((95, 95), got)
        self.assertEqual(len(got), 2)

    def test_query_half_open_boundary(self):
        # Point on the query's right/bottom edge is excluded.
        t = QuadTree(0, 0, 100, 100, capacity=4)
        t.insert(10, 10)
        t.insert(20, 10)
        got = t.query(0, 0, 10, 100)  # x in [0,10)
        self.assertEqual(got, [])  # (10,10) excluded; (20,10) outside x range
        got = t.query(0, 0, 20, 100)  # x in [0,20)
        self.assertEqual(got, [(10, 10)])  # (20,10) excluded

    def test_query_invalid_box_returns_empty(self):
        t = QuadTree(0, 0, 100, 100)
        t.insert(5, 5)
        self.assertEqual(t.query(50, 50, 10, 10), [])  # x1 < x0
        self.assertEqual(t.query(10, 50, 100, 10), [])  # y1 < y0
        self.assertEqual(t.query(10, 10, 10, 10), [])   # zero-area

    def test_query_after_many_inserts(self):
        t = QuadTree(0, 0, 100, 100, capacity=4, max_depth=8)
        pts = []
        for i in range(50):
            x = float(i)
            y = float(i * 2)
            t.insert(x, y)
            pts.append((x, y))
        got = set(t.query(0, 0, 100, 100))
        self.assertEqual(got, set(pts))


class TestExports(unittest.TestCase):

    def test_point_alias_is_tuple(self):
        # Point is a typing alias for Tuple[float, float]; plain tuples work
        # as points end-to-end. We verify importability and behavior.
        t = QuadTree(0, 0, 10, 10)
        t.insert(1, 2)
        self.assertEqual(t.query(0, 0, 10, 10), [(1, 2)])
        self.assertTrue(Point is not None)

    def test_invalid_capacity_raises(self):
        with self.assertRaises(ValueError):
            QuadTree(0, 0, 10, 10, capacity=0)
        with self.assertRaises(ValueError):
            QuadTree(0, 0, 10, 10, capacity=-1)

    def test_invalid_max_depth_raises(self):
        with self.assertRaises(ValueError):
            QuadTree(0, 0, 10, 10, max_depth=0)


if __name__ == "__main__":
    unittest.main()
