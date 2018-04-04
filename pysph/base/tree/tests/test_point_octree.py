import numpy as np

from pysph.base.tree.point_octree import OctreeGPU
from pysph.base.opencl import DeviceHelper
from pysph.base.utils import get_particle_array

from pysph.base.nnps_base import NNPSParticleArrayWrapper

import unittest


def _gen_uniform_dataset(n, h, seed=None):
    if seed is not None:
        np.random.seed(seed)
    u = np.random.uniform
    pa = get_particle_array(x=u(size=n), y=u(size=n), z=u(size=n), h=h)
    h = DeviceHelper(pa)
    pa.set_device_helper(h)

    return pa


def _dfs_find_leaf(octree):
    leaf_id_count = octree.allocate_leaf_prop(np.int32)
    dfs_find_leaf = octree.leaf_tree_traverse(
        "int *leaf_id_count",
        setup="leaf_id_count[i] = 0;",
        node_operation="if (cid_dst == cid_src) leaf_id_count[i]++",
        leaf_operation="if (cid_dst == cid_src) leaf_id_count[i]++",
        output_expr=""
    )

    dfs_find_leaf(octree, octree, leaf_id_count.array)
    return leaf_id_count.array.get()


def _check_children_overlap(node_xmin, node_xmax, child_offset):
    for j in range(8):
        nxmin1 = node_xmin[child_offset + j]
        nxmax1 = node_xmax[child_offset + j]
        for k in range(8):
            nxmin2 = node_xmin[child_offset + k]
            nxmax2 = node_xmax[child_offset + k]
            if j != k:
                assert (nxmax1[0] <= nxmin2[0] or nxmax2[0] <= nxmin1[0] or
                        nxmax1[1] <= nxmin2[1] or nxmax2[1] <= nxmin1[1] or
                        nxmax1[2] <= nxmin2[2] or nxmax2[2] <= nxmin1[2])


class OctreeTestCase(unittest.TestCase):
    def setUp(self):
        use_double = False
        self.N = 3000
        pa = _gen_uniform_dataset(self.N, 0.2, seed=0)
        self.octree = OctreeGPU(pa, radius_scale=1., use_double=use_double,
                                leaf_size=64)
        self.octree.refresh(np.array([0., 0., 0.]), np.array([1., 1., 1.]),
                            np.min(pa.h))
        self.pa = pa

    def test_pids(self):
        pids = self.octree.pids.array.get()
        s = set()
        for i in range(len(pids)):
            if 0 <= pids[i] < self.N:
                s.add(pids[i])

        assert (len(s) == self.N)

    def test_depth_and_inclusiveness(self):
        """
        Traverse tree and check if max depth is correct
        Additionally check if particle sets of siblings is disjoint
        and union of particle sets of a nodes children = nodes own children
        :return:
        """
        s = [0, ]
        d = [0, ]

        offsets = self.octree.offsets.array.get()
        pbounds = self.octree.pbounds.array.get()

        max_depth = self.octree.depth
        max_depth_here = 0
        pids = set()

        while len(s) != 0:
            n = s[0]
            depth = d[0]
            max_depth_here = max(max_depth_here, depth)
            pbound = pbounds[n]
            assert (depth <= max_depth)

            del s[0]
            del d[0]

            if offsets[n] == -1:
                # assert (pbounds[n][1] - pbounds[n][0] <= 32)
                for i in range(pbound[0], pbound[1]):
                    pids.add(i)
                continue

            # Particle ranges of children are contiguous
            # and are contained within parent's particle arange
            l = pbound[0]
            for i in range(8):
                child_idx = offsets[n] + i
                assert (pbounds[child_idx][0] == l)
                assert (pbounds[child_idx][0] <= pbounds[child_idx][1])
                l = pbounds[child_idx][1]

                assert (child_idx < len(offsets))
                s.append(child_idx)
                d.append(depth + 1)
            assert (l == pbound[1])

    def test_node_bounds(self):
        # TODO: Add test to check overlapping

        self.octree._set_node_bounds()
        pids = self.octree.pids.array.get()
        offsets = self.octree.offsets.array.get()
        pbounds = self.octree.pbounds.array.get()
        node_xmin = self.octree.node_xmin.array.get()
        node_xmax = self.octree.node_xmax.array.get()
        node_hmax = self.octree.node_hmax.array.get()

        x = self.pa.x[pids]
        y = self.pa.y[pids]
        z = self.pa.z[pids]
        for i in range(len(offsets)):
            nxmin = node_xmin[i]
            nxmax = node_xmax[i]

            for j in range(pbounds[i][0], pbounds[i][1]):
                assert (nxmin[0] <= np.float32(x[j]) <= nxmax[0])
                assert (nxmin[1] <= np.float32(y[j]) <= nxmax[1])
                assert (nxmin[2] <= np.float32(z[j]) <= nxmax[2])

            # Check that children nodes don't overlap
            if offsets[i] != -1:
                _check_children_overlap(node_xmin, node_xmax, offsets[i])

    def test_dfs_traversal(self):
        leaf_id_count = _dfs_find_leaf(self.octree)
        np.testing.assert_array_equal(
            np.ones(self.octree.unique_cid_count, dtype=np.int32),
            leaf_id_count
        )

    def tearDown(self):
        del self.octree
