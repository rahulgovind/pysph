import numpy as np

from pysph.base.tree.point_octree import OctreeGPU
from pysph.base.opencl import DeviceHelper, DeviceArray
from pysph.base.utils import get_particle_array

from pysph.base.nnps_base import NNPSParticleArrayWrapper
from pysph.base.tests.test_nnps import NNPSTestCase

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


def _test_nnps(neighbor_count, pa1, pa2, pids=None, sorted=False, radius_scale=1.):
    n1 = pa1.get_number_of_particles()
    n2 = pa2.get_number_of_particles()

    xd, yd, zd, hd = np.array(pa1.x), np.array(pa1.y), \
                     np.array(pa1.z), np.array(pa1.h)
    xs, ys, zs, hs = np.array(pa2.x), np.array(pa2.y), \
                     np.array(pa2.z), np.array(pa2.h)

    count = np.zeros(n1, dtype=int)

    for i in range(n1):
        for j in range(n2):
            dist2 = (xd[i] - xs[j]) ** 2 + \
                    (yd[i] - ys[j]) ** 2 + (zd[i] - zs[j]) ** 2
            if dist2 < hd[i] ** 2 * radius_scale ** 2 or dist2 < hs[j] ** 2 * radius_scale ** 2:
                count[i] += 1

    if sorted:
        pids_inv = np.zeros(n, dtype=int)
        for i in range(n1):
            pids_inv[pids[i]] = i
    else:
        pids_inv = np.arange(n1)

    for i in range(n1):
        assert (count[i] == neighbor_count[pids_inv[i]])


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
        # TODO: Add test to check h

        self.octree._set_node_bounds()
        pids = self.octree.pids.array.get()
        offsets = self.octree.offsets.array.get()
        pbounds = self.octree.pbounds.array.get()
        node_xmin = self.octree.node_xmin.array.get()
        node_xmax = self.octree.node_xmax.array.get()

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


class OctreeNNPSTestCase(NNPSTestCase):
    def _find_domain(self, pa_idx):
        pa = self.particles[pa_idx]
        xmin = np.array([np.min(pa.x), np.min(pa.y), np.min(pa.z)])
        xmax = np.array([np.max(pa.x), np.max(pa.y), np.max(pa.z)])
        hmin = np.min(pa.h)
        return xmin, xmax, hmin

    def setUp(self):
        super(OctreeNNPSTestCase, self).setUp()

        for i in range(len(self.particles)):
            pa = self.particles[i]
            h = DeviceHelper(pa)
            pa.set_device_helper(h)

        self.octrees = [
            OctreeGPU(self.particles[i], radius_scale=2.)
            for i in range(len(self.particles))
            ]

        for i in range(len(self.particles)):
            xmin, xmax, hmin = self._find_domain(i)
            self.octrees[i].refresh(xmin - hmin / 2, xmax + hmin / 2, hmin)
            self.octrees[i]._set_node_bounds()


    def test_depth_and_inclusiveness(self):
        """
        Traverse tree and check if max depth is correct
        Additionally check if particle sets of siblings is disjoint
        and union of particle sets of a nodes children = nodes own children
        :return:
        """
        for octree in self.octrees:
            s = [0, ]
            d = [0, ]

            offsets = octree.offsets.array.get()
            pbounds = octree.pbounds.array.get()

            max_depth = octree.depth
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
        # TODO: Add test to check h

        for k, octree in enumerate(self.octrees):
            octree._set_node_bounds()
            pids = octree.pids.array.get()
            offsets = octree.offsets.array.get()
            pbounds = octree.pbounds.array.get()
            node_xmin = octree.node_xmin.array.get()
            node_xmax = octree.node_xmax.array.get()
            pa = self.particles[k]
            x = pa.x[pids]
            y = pa.y[pids]
            z = pa.z[pids]
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
        for octree in self.octrees:
            leaf_id_count = _dfs_find_leaf(octree)
            np.testing.assert_array_equal(
                np.ones(octree.unique_cid_count, dtype=np.int32),
                leaf_id_count
            )


    def _test_neighbors_by_particle(self, src_index, dst_index):
        octree_dst = self.octrees[dst_index]
        octree_src = self.octrees[src_index]
        nbr_cid_lengths, nbr_cids = octree_dst._find_neighbor_cids(octree_src)
        n = self.particles[dst_index].get_number_of_particles()
        neighbor_count = DeviceArray(np.uint32, n)
        octree_dst._find_neighbor_lengths(nbr_cid_lengths, nbr_cids,
                                          octree_src, neighbor_count.array)
        _test_nnps(neighbor_count.array.get(),
                   self.particles[dst_index],
                   self.particles[src_index], radius_scale=2.)

    def test_neighbors_aa(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=0)

    def test_neighbors_ab(self):
        self._test_neighbors_by_particle(src_index=0, dst_index=1)

    def test_neighbors_ba(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=0)

    def test_neighbors_bb(self):
        self._test_neighbors_by_particle(src_index=1, dst_index=1)

    def test_neighbors_cc(self):
        self._test_neighbors_by_particle(src_index=2, dst_index=2)

    def test_neighbors_dd(self):
        self._test_neighbors_by_particle(src_index=3, dst_index=3)
