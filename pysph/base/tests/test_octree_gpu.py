from unittest import TestCase
import pytest
import numpy as np

pytest.importorskip("pysph.base.opencl")

from pysph.base.octree_gpu import OctreeGPU


class TestOctreeGPU(TestCase):
    def setUp(self):
        pass

    def test_calc_cell_size_and_depth(self):
        pass

    def test_bin(self):
        pass


