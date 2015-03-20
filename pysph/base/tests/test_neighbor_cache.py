"""Tests for neighbor caching.
"""

import numpy as np
import unittest

from pysph.base.nnps import LinkedListNNPS, NeighborCache
from pysph.base.utils import get_particle_array
from pyzoltan.core.carray import UIntArray

class TestNeighborCache(unittest.TestCase):
    def _make_random_parray(self, name, nx=5):
        x, y, z = np.random.random((3, nx, nx, nx))
        x = np.ravel(x)
        y = np.ravel(y)
        z = np.ravel(z)
        h = np.ones_like(x)*0.2
        return get_particle_array(name=name, x=x, y=y, z=z, h=h)

    def test_neighbors_cached_properly(self):
        # Given
        pa1 = self._make_random_parray('pa1', 5)
        pa2 = self._make_random_parray('pa2', 4)
        particles = [pa1, pa2]
        nnps = LinkedListNNPS(dim=3, particles=particles)

        for dest_index in (0, 1):

            # When
            cache = NeighborCache(nnps, dest_index)
            cache.update()
            nb_cached = UIntArray()
            nb_direct = UIntArray()

            # Then.
            for src_idx in (0, 1):
                for i in range(len(particles[dest_index].x)):
                    nnps.get_nearest_particles(src_idx, dest_index, i, nb_direct)
                    cache.get_neighbors(src_idx, i, nb_cached)
                    nb_e = nb_direct.get_npy_array()
                    nb_c = nb_cached.get_npy_array()
                    self.assertTrue(np.all(nb_e == nb_c))

    def test_cache_updates_with_changed_particles(self):
        # Given
        pa1 = self._make_random_parray('pa1', 5)
        particles = [pa1]
        nnps = LinkedListNNPS(dim=3, particles=particles)
        cache = NeighborCache(nnps, 0)
        cache.update()

        # When
        pa2 = self._make_random_parray('pa2', 2)
        pa1.add_particles(x=pa2.x, y=pa2.y, z=pa2.z)
        nnps.update()
        cache.update()
        nb_cached = UIntArray()
        nb_direct = UIntArray()
        for i in range(len(particles[0].x)):
            nnps.get_nearest_particles(0, 0, i, nb_direct)
            cache.get_neighbors(0, i, nb_cached)
            nb_e = nb_direct.get_npy_array()
            nb_c = nb_cached.get_npy_array()
            self.assertTrue(np.all(nb_e == nb_c))


if __name__ == '__main__':
    unittest.main()
