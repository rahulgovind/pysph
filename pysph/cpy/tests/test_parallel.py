from math import sin
import unittest
import numpy as np

from pytest import importorskip

from ..config import get_config, use_config
from ..array import wrap
from ..types import annotate
from ..parallel import Elementwise, Reduction, Scan


class TestParallelUtils(unittest.TestCase):
    def setUp(self):
        cfg = get_config()
        self._use_double = cfg.use_double
        cfg.use_double = True

    def tearDown(self):
        get_config().use_double = self._use_double

    def _check_simple_elementwise(self, backend):
        # Given
        @annotate(i='int', x='doublep', y='doublep', double='a,b')
        def axpb(i, x, y, a, b):
            y[i] = a * sin(x[i]) + b

        x = np.linspace(0, 1, 10000)
        y = np.zeros_like(x)
        a = 2.0
        b = 3.0
        x, y = wrap(x, y, backend=backend)

        # When
        e = Elementwise(axpb, backend=backend)
        e(x, y, a, b)

        # Then
        y.pull()
        self.assertTrue(np.allclose(y.data, a * np.sin(x.data) + b))

    def test_elementwise_works_with_cython(self):
        self._check_simple_elementwise(backend='cython')

    def test_elementwise_works_with_opencl(self):
        importorskip('pyopencl')

        self._check_simple_elementwise(backend='opencl')

    def test_elementwise_works_with_cuda(self):
        importorskip('pycuda')

        self._check_simple_elementwise(backend='cuda')

    def _check_simple_reduction(self, backend):
        x = np.linspace(0, 1, 1000) / 1000
        x = wrap(x, backend=backend)

        # When
        r = Reduction('a+b', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def _check_reduction_min(self, backend):
        x = np.linspace(0, 1, 1000) / 1000
        x = wrap(x, backend=backend)

        # When
        r = Reduction('min(a, b)', neutral='INFINITY', backend=backend)
        result = r(x)

        # Then
        self.assertAlmostEqual(result, 0.0, 6)

    def _check_reduction_with_map(self, backend):
        # Given
        from math import cos, sin
        x = np.linspace(0, 1, 1000) / 1000
        y = x.copy()
        x, y = wrap(x, y, backend=backend)

        @annotate(i='int', doublep='x, y')
        def map(i=0, x=[0.0], y=[0.0]):
            return cos(x[i]) * sin(y[i])

        # When
        r = Reduction('a+b', map_func=map, backend=backend)
        result = r(x, y)

        # Then
        self.assertAlmostEqual(result, 0.5, 6)

    def test_reduction_works_without_map_cython(self):
        self._check_simple_reduction(backend='cython')

    def test_reduction_works_with_map_cython(self):
        self._check_reduction_with_map(backend='cython')

    def test_reduction_works_neutral_cython(self):
        self._check_reduction_min(backend='cython')

    def test_reduction_works_without_map_opencl(self):
        importorskip('pyopencl')
        self._check_simple_reduction(backend='opencl')

    def test_reduction_works_with_map_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_with_map(backend='opencl')

    def test_reduction_works_neutral_opencl(self):
        importorskip('pyopencl')
        self._check_reduction_min(backend='opencl')

    def test_reduction_works_without_map_cuda(self):
        importorskip('pycuda')
        self._check_simple_reduction(backend='cuda')

    def test_reduction_works_with_map_cuda(self):
        importorskip('pycuda')
        self._check_reduction_with_map(backend='cuda')

    def test_reduction_works_neutral_cuda(self):
        importorskip('pycuda')
        self._check_reduction_min(backend='cuda')

    def test_scan_works_opencl(self):
        importorskip('pyopencl')
        # Given
        a = np.arange(10000, dtype=np.int32)
        data = a.copy()
        a = wrap(a, backend='opencl')

        @annotate(i='int', ary='intp', return_='int')
        def input_f(i, ary):
            return ary[i]

        @annotate(int='i, item', ary='intp')
        def output_f(i, ary, item):
            ary[i + 1] = item

        # When
        s = Scan(input_f, output_f, 'a+b', dtype=np.int32, backend='opencl')
        s(a)
        a.pull()
        result = a.data

        # Then
        expect = np.cumsum(data)
        # print(result, y)
        self.assertTrue(np.all(expect[:-1] == result[1:]))

    def _get_scan_parameters(self, backend):
        a = np.arange(10000, dtype=np.int32)
        data = a.copy()
        expect = np.cumsum(data)

        a = wrap(a, backend=backend)

        @annotate(i='int', input_ary='intp', return_='int')
        def input_f(i, input_ary):
            return input_ary[i]

        @annotate(int='i, item', output_ary='intp')
        def output_f(i, item, output_ary):
            output_ary[i] = item

        scan = Scan(input_f, output_f, 'a+b', dtype=np.int32,
                    backend=backend)

        return scan, a, expect

    def test_scan_works_openmp(self):
        scan, a, expect = self._get_scan_parameters(backend='cython')

        scan(a, a)
        a.pull()
        result = a.data

        np.testing.assert_equal(expect, result)

    def test_scan_works_openmp_parallel(self):
        with use_config(use_openmp=True):
            scan, a, expect = self._get_scan_parameters(backend='cython')

            scan(a, a)
            a.pull()
            result = a.data

            np.testing.assert_equal(expect, result)

    def _get_unique_scan_parameters(self, backend):
        a = np.random.randint(0, 100, 100, dtype=np.int32)
        a = np.sort(a)
        data = a.copy()

        unique_ary_actual = np.sort(np.unique(data))
        unique_count_actual = len(np.unique(data))

        a = wrap(a, backend=backend)

        unique_ary = np.zeros(len(a.data), dtype=np.int32)
        unique_ary = wrap(unique_ary, backend=backend)

        unique_count = np.zeros(1, dtype=np.int32)
        unique_count = wrap(unique_count, backend=backend)

        @annotate(i='int', input_ary='intp', return_='int')
        def input_f(i, input_ary):
            if i == 0 or input_ary[i] != input_ary[i - 1]:
                return 1
            else:
                return 0

        @annotate(int='i, item', ary2='intp')
        def output_f(i, item, ary2):
            ary2[i] = item

        @annotate(int='i, prev_item, item, N', ary2='intp',
                  unique='intp', unique_count='intp')
        def output_f(i, prev_item, item, N, ary2, unique, unique_count):
            if item != prev_item:
                unique[item - 1] = ary2[i]
            if i == N - 1:
                unique_count[0] = item

        # When
        s = Scan(input_f, output_f, 'a+b', dtype=np.int32, backend=backend)
        return (s, a, unique_ary, unique_count, unique_ary_actual,
                unique_count_actual)

    def test_scan_works_openmp_with_unique(self):
        # Given
        (scan, a, unique_ary, unique_count, unique_ary_actual,
         unique_count_actual) = \
            self._get_unique_scan_parameters(backend='cython')

        # When
        scan(a, a, unique_ary, unique_count)
        unique_ary.pull()
        unique_count.pull()

        unique_count = unique_count.data[0]

        # Then
        np.testing.assert_equal(unique_ary_actual,
                                unique_ary.data[:unique_count])
        assert (unique_count == unique_count_actual)

    def test_scan_works_openmp_with_unique_parallel(self):
        with use_config(use_openmp=True):
            (scan, a, unique_ary, unique_count, unique_ary_actual,
             unique_count_actual) = \
                self._get_unique_scan_parameters(backend='cython')

            scan(a, a, unique_ary, unique_count)
            unique_ary.pull()
            unique_count.pull()

            unique_count = unique_count.data[0]

            np.testing.assert_equal(unique_ary_actual,
                                    unique_ary.data[:unique_count])
            assert (unique_count == unique_count_actual)

    def _get_segmented_scan_actual(self, a, segment_flags):
        output_actual = np.zeros_like(a)
        for i in range(len(a)):
            if segment_flags[i] == 0 and i != 0:
                output_actual[i] = output_actual[i - 1] + a[i]
            else:
                output_actual[i] = a[i]
        return output_actual

    def _get_segmented_scan_parameters(self, backend):
        # Given
        a = np.random.randint(0, 100, 50000, dtype=np.int32)
        a_copy = a.copy()

        seg = np.random.randint(0, 100, 50000, dtype=np.int32)
        seg = (seg == 0).astype(np.int32)
        seg_copy = seg.copy()

        a = wrap(a, backend=backend)
        seg = wrap(seg, backend=backend)

        @annotate(i='int', input_ary='intp', return_='int')
        def input_f(i, input_ary):
            return input_ary[i]

        @annotate(i='int', seg_flag='intp', return_='int')
        def segment_f(i, seg_flag):
            return seg_flag[i]

        @annotate(int='i, item', output_ary='intp')
        def output_f(i, item, output_ary):
            output_ary[i] = item

        scan = Scan(input_f, output_f, 'a+b', dtype=np.int32, backend=backend,
                    is_segment=segment_f)

        output_actual = self._get_segmented_scan_actual(a_copy, seg_copy)
        return scan, a, seg, output_actual

    def test_segmented_scan_cython(self):
        scan, a, seg, output_actual = self._get_segmented_scan_parameters(
            backend='cython')

        scan(a, seg, a)
        a.pull()

        np.testing.assert_equal(output_actual, a.data)

    def test_segmented_scan_cython_parallel(self):
        with use_config(use_openmp=True):
            scan, a, seg, output_actual = \
                self._get_segmented_scan_parameters(backend='cython')

            scan(a, seg, a)
            a.pull()

            np.testing.assert_equal(output_actual, a.data)
