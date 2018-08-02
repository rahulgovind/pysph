from pysph.base.tree.tree import Tree
from pysph.base.tree.helpers import ParticleArrayWrapper, get_helper, \
    make_vec_dict, ctype_to_dtype, get_vector_dtype
from pysph.base.opencl import profile_kernel, DeviceArray, \
    DeviceWGSException, get_queue
from pysph.cpy.opencl import named_profile
from pytools import memoize

import sys
import numpy as np

import pyopencl as cl
import pyopencl.cltypes
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.scan import GenericScanKernel

from mako.template import Template

# For Mako
disable_unicode = False if sys.version_info.major > 2 else True


class IncompatibleOctreesException(Exception):
    pass


@named_profile('neighbor_psum')
@memoize
def _get_neighbor_psum_kernel(ctx):
    return GenericScanKernel(
        ctx, np.int32, neutral="0",
        arguments=r"""__global int *neighbor_counts""",
        input_expr="neighbor_counts[i]",
        scan_expr="a + b",
        output_statement=r"""neighbor_counts[i] = prev_item;"""
    )


@memoize
def _get_macros_preamble(c_type, sorted):
    return Template("""
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
    #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
    #define AVG(X, Y) (((X) + (Y)) / 2)
    #define ABS(X) ((X) > 0 ? (X) : -(X))
    #define EPS 1e-6f
    #define INF 1e6
    #define SQR(X) ((X) * (X))

    % if sorted:
    #define PID(idx) (idx)
    % else:
    #define PID(idx) (pids[idx])
    % endif

    char contains(private ${data_t} *n1, private ${data_t} *n2) {
        // Check if node n1 contains node n2
        char res = 1;
        %for i in range(3):
            res = res && (n1[${i}] <= n2[${i}]) &&
                  (n1[3 + ${i}] >= n2[3 + ${i}]);
        %endfor

        return res;
    }

    char contains_search(private ${data_t} *n1,
                         private ${data_t} *n2) {
        // Check if node n1 contains node n2 with n1 having
        // its search radius extension
        ${data_t} h = n1[6];
        char res = 1;
        %for i in range(3):
            res = res & (n1[${i}] - h - EPS <= n2[${i}]) &
                  (n1[3 + ${i}] + h + EPS >= n2[3 + ${i}]);
        %endfor

        return res;
    }

    char intersects(private ${data_t} *n1, private ${data_t} *n2) {
        // Check if node n1 'intersects' node n2
        ${data_t} cdist;
        ${data_t} w1, w2, wavg = 0;
        char res = 1;
        ${data_t} h = MAX(n1[6], n2[6]);

        % for i in range(3):
            cdist = fabs((n1[${i}] + n1[3 + ${i}]) / 2 -
                         (n2[${i}] + n2[3 + ${i}]) / 2);
            w1 = fabs(n1[${i}] - n1[3 + ${i}]);
            w2 = fabs(n2[${i}] - n2[3 + ${i}]);
            wavg = AVG(w1, w2);
            res &= (cdist - wavg <= h + EPS);
        % endfor

        return res;
    }
    """, disable_unicode=disable_unicode).render(data_t=c_type, sorted=sorted)


@memoize
def _get_node_bound_kernel_parameters(dim, data_t, xvars):
    result = {}
    result['setup'] = Template(
        r"""
        ${data_t} xmin[${dim}] = {${', '.join(['1e6'] * dim)}};
        ${data_t} xmax[${dim}] = {${', '.join(['-1e6'] * dim)}};
        ${data_t} hmax = 0;
        """).render(dim=dim, data_t=data_t)

    result['args'] = Template(
        r"""int *pids,
        % for v in xvars:
        ${data_t} *${v},
        % endfor
        ${data_t} *h,
        ${data_t} radius_scale,
        ${data_t}${dim} *node_xmin,
        ${data_t}${dim} *node_xmax,
        ${data_t} *node_hmax
        """).render(dim=dim, data_t=data_t, xvars=xvars)

    result['leaf_operation'] = Template(
        r"""
        for (int j=pbound.s0; j < pbound.s1; j++) {
        int pid = PID(j);
        % for d in range(dim):
            xmin[${d}] = fmin(xmin[${d}], ${xvars[d]}[pid] - 1e-6f);
            xmax[${d}] = fmax(xmax[${d}], ${xvars[d]}[pid] + 1e-6f);
        % endfor
        hmax = fmax(h[pid] * radius_scale, hmax);
        }
        """).render(dim=dim, xvars=xvars)

    result['node_operation'] = Template(
        r"""
        % for i in range(2 ** dim):
            % for d in range(dim):
                xmin[${d}] = fmin(
                    xmin[${d}], node_xmin[child_offset + ${i}].s${d}
                );
                xmax[${d}] = fmax(
                    xmax[${d}], node_xmax[child_offset + ${i}].s${d}
                );
            % endfor
            hmax = fmax(hmax, node_hmax[child_offset + ${i}]);
        % endfor
        """).render(dim=dim)

    result['output_expr'] = Template(
        """
        % for d in range(dim):
            node_xmin[node_idx].s${d} = xmin[${d}];
            node_xmax[node_idx].s${d} = xmax[${d}];
        % endfor
        node_hmax[node_idx] = hmax;
        """).render(dim=dim)

    return result


@memoize
def _get_leaf_neighbor_kernel_parameters(data_t, args, setup, operation,
                                         output_expr):
    result = {
        'setup': Template(r"""
            ${data_t} ndst[8];
            ${data_t} nsrc[8];

            %% for i in range(3):
                ndst[${i}] = node_xmin_dst[cid_dst].s${i};
                ndst[${i} + 3] = node_xmax_dst[cid_dst].s${i};
            %% endfor
            ndst[6] = node_hmax_dst[cid_dst];

            %(setup)s;
            """ % dict(setup=setup)).render(data_t=data_t),
        'node_operation': Template("""
            % for i in range(3):
                nsrc[${i}] = node_xmin_src[cid_src].s${i};
                nsrc[${i} + 3] = node_xmax_src[cid_src].s${i};
            % endfor
            nsrc[6] = node_hmax_src[cid_src];

            if (!intersects(ndst, nsrc) && !contains(nsrc, ndst)) {
                flag = 0;
                break;
            }
            """).render(data_t=data_t),
        'leaf_operation': Template("""
            %% for i in range(3):
                nsrc[${i}] = node_xmin_src[cid_src].s${i};
                nsrc[${i} + 3] = node_xmax_src[cid_src].s${i};
            %% endfor
            nsrc[6] = node_hmax_src[cid_src];

            if (intersects(ndst, nsrc) || contains_search(ndst, nsrc)) {
                %(operation)s;
            }
            """ % dict(operation=operation)).render(),
        'output_expr': output_expr,
        'args': Template("""
            ${data_t}3 *node_xmin_src, ${data_t}3 *node_xmax_src,
            ${data_t} *node_hmax_src,
            ${data_t}3 *node_xmin_dst, ${data_t}3 *node_xmax_dst,
            ${data_t} *node_hmax_dst,
            """ + args).render(data_t=data_t)
    }
    return result


class PointTree(Tree):
    def __init__(self, pa, dim=2, leaf_size=32, radius_scale=2.0,
                 use_double=False, c_type='float'):
        super(PointTree, self).__init__(pa.get_number_of_particles(), 2 ** dim,
                                        leaf_size)

        assert (1 <= dim <= 3)
        self.max_depth = None
        self.dim = dim
        self.powdim = 2 ** self.dim
        self.xvars = ('x', 'y', 'z')[:dim]

        self.c_type = c_type
        self.c_type_src = 'double' if use_double else 'float'
        self.pa = ParticleArrayWrapper(pa, self.c_type_src,
                                       self.c_type, self.xvars + ('h',))

        self.radius_scale = radius_scale
        self.use_double = use_double

        self.helper = get_helper(self.ctx, 'tree/point_tree.mako', self.c_type)
        self.xmin = None
        self.xmax = None
        self.hmin = None
        self.make_vec = make_vec_dict[c_type][self.dim]

    def set_vars(self):
        self.data_vars = ["sfc"]
        self.data_var_ctypes = ["ulong"]
        self.data_var_dtypes = [np.uint64]
        self.const_vars = ['mask', 'rshift']
        self.const_var_ctypes = ['ulong', 'char']
        self.index_code = "((sfc[i] & mask) >> rshift)"

    def _calc_cell_size_and_depth(self):
        self.cell_size = self.hmin * self.radius_scale * (1. + 1e-3)
        self.cell_size /= 128
        max_width = max((self.xmax[i] - self.xmin[i]) for i in range(self.dim))
        self.max_depth = int(np.ceil(np.log2(max_width / self.cell_size))) + 1

    def _bin(self):
        dtype = ctype_to_dtype(self.c_type)
        fill_particle_data = self.helper.get_kernel("fill_particle_data",
                                                    dim=self.dim,
                                                    xvars=self.xvars)
        pa_gpu = self.pa.gpu
        args = [getattr(pa_gpu, v) for v in self.xvars]
        args += [dtype(self.cell_size),
                 self.make_vec(*[self.xmin[i] for i in range(self.dim)]),
                 self.sfc.array, self.pids.array]
        fill_particle_data(*args)

    def get_index_constants(self, depth):
        rshift = np.uint8(self.dim * (self.max_depth - depth - 1))
        mask = np.uint64((2 ** self.dim - 1) << rshift)
        return mask, rshift

    def _adjust_domain_width(self):
        # Convert width of domain to a power of 2 multiple of cell size
        # (Optimal width for cells)
        cell_size = self.hmin * self.radius_scale * (1. + 1e-5)
        max_width = np.max(self.xmax - self.xmin)

        new_width = cell_size * \
                    2.0 ** int(np.ceil(np.log2(max_width / cell_size)))

        diff = (new_width - (self.xmax - self.xmin)) / 2

        self.xmin -= diff
        self.xmax += diff

    def setup_build(self, xmin, xmax, hmin):
        self._setup_build()
        self.pa.sync()
        self.xmin = np.array(xmin)
        self.xmax = np.array(xmax)
        self.hmin = hmin
        self._adjust_domain_width()
        self._calc_cell_size_and_depth()
        self._bin()

    def build(self, fixed_depth=None):
        self._build(self.max_depth if fixed_depth is None else fixed_depth)
        self._get_unique_cids_and_count()

    def refresh(self, xmin, xmax, hmin, fixed_depth=None):
        self.setup_build(xmin, xmax, hmin)
        self.build(fixed_depth)

    def _sort(self):
        """Set octree as being sorted

        The particle array needs to be aligned by the caller!
        """
        if not self.sorted:
            self.pa.force_sync()
            self.sorted = 1

    ###########################################################################
    # Algos
    ###########################################################################
    def set_node_bounds(self):
        vdata_t = get_vector_dtype(self.c_type, self.dim)
        data_t = self.c_type

        self.node_xmin = self.allocate_node_prop(vdata_t)
        self.node_xmax = self.allocate_node_prop(vdata_t)
        self.node_hmax = self.allocate_node_prop(data_t)

        params = _get_node_bound_kernel_parameters(self.dim, self.c_type,
                                                   self.xvars)
        set_node_bounds = self.tree_bottom_up(
            params['args'], params['setup'], params['leaf_operation'],
            params['node_operation'], params['output_expr'],
            preamble=_get_macros_preamble(self.c_type, self.sorted)
        )
        set_node_bounds = profile_kernel(set_node_bounds, 'set_node_bounds')

        pa_gpu = self.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        args = [self, self.pids.array]
        args += [getattr(pa_gpu, v) for v in self.xvars]
        args += [pa_gpu.h,
                 dtype(self.radius_scale),
                 self.node_xmin.array, self.node_xmax.array,
                 self.node_hmax.array]

        set_node_bounds(*args)

    ###########################################################################
    # Nearest Neighbor Particle Search (NNPS)
    ###########################################################################
    def _leaf_neighbor_operation(self, octree_src, args, setup, operation,
                                 output_expr):
        """
        Template for finding neighboring cids of a cell.
        """
        params = _get_leaf_neighbor_kernel_parameters(self.c_type, args,
                                                      setup, operation,
                                                      output_expr)
        kernel = octree_src.leaf_tree_traverse(
            params['args'], params['setup'], params['node_operation'],
            params['leaf_operation'], params['output_expr'],
            preamble=_get_macros_preamble(self.c_type, self.sorted)
        )

        def callable(*args):
            return kernel(octree_src, self,
                          octree_src.node_xmin.array,
                          octree_src.node_xmax.array,
                          octree_src.node_hmax.array,
                          self.node_xmin.array, self.node_xmax.array,
                          self.node_hmax.array,
                          *args)

        return callable

    def find_neighbor_cids(self, octree_src):
        neighbor_cid_count = DeviceArray(np.uint32, self.unique_cid_count + 1)
        find_neighbor_cid_counts = self._leaf_neighbor_operation(
            octree_src,
            args="uint2 *pbounds, int *cnt",
            setup="int count=0",
            operation="""
                    if (pbounds[cid_src].s0 < pbounds[cid_src].s1)
                        count++;
                    """,
            output_expr="cnt[i] = count;"
        )
        find_neighbor_cid_counts = profile_kernel(
            find_neighbor_cid_counts, 'find_neighbor_cid_count')
        find_neighbor_cid_counts(octree_src.pbounds.array,
                                 neighbor_cid_count.array)

        neighbor_psum = _get_neighbor_psum_kernel(self.ctx)
        neighbor_psum(neighbor_cid_count.array)

        total_neighbors = int(neighbor_cid_count.array[-1].get())
        neighbor_cids = DeviceArray(np.uint32, total_neighbors)

        find_neighbor_cids = self._leaf_neighbor_operation(
            octree_src,
            args="uint2 *pbounds, int *cnt, int *neighbor_cids",
            setup="int offset=cnt[i];",
            operation="""
            if (pbounds[cid_src].s0 < pbounds[cid_src].s1)
                neighbor_cids[offset++] = cid_src;
            """,
            output_expr=""
        )
        find_neighbor_cids = profile_kernel(
            find_neighbor_cids, 'find_neighbor_cids')
        find_neighbor_cids(octree_src.pbounds.array,
                           neighbor_cid_count.array, neighbor_cids.array)
        return neighbor_cid_count, neighbor_cids

    def find_neighbor_lengths_elementwise(self, neighbor_cid_count,
                                          neighbor_cids, octree_src,
                                          neighbor_count):
        self.check_nnps_compatibility(octree_src)

        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        find_neighbor_counts = self.helper.get_kernel(
            'find_neighbor_counts_elementwise', sorted=self.sorted
        )
        find_neighbor_counts(self.unique_cids_map.array, octree_src.pids.array,
                             self.pids.array,
                             self.cids.array,
                             octree_src.pbounds.array, self.pbounds.array,
                             pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z,
                             pa_gpu_src.h,
                             pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z,
                             pa_gpu_dst.h,
                             dtype(self.radius_scale),
                             neighbor_cid_count.array,
                             neighbor_cids.array,
                             neighbor_count)

    def find_neighbors_elementwise(self, neighbor_cid_count, neighbor_cids,
                                   octree_src, start_indices, neighbors):
        self.check_nnps_compatibility(octree_src)

        n = self.pa.get_number_of_particles()
        wgs = self.leaf_size
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu

        dtype = ctype_to_dtype(self.c_type)

        find_neighbors = self.helper.get_kernel(
            'find_neighbors_elementwise', sorted=self.sorted)
        find_neighbors(self.unique_cids_map.array, octree_src.pids.array,
                       self.pids.array,
                       self.cids.array,
                       octree_src.pbounds.array, self.pbounds.array,
                       pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z, pa_gpu_src.h,
                       pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z, pa_gpu_dst.h,
                       dtype(self.radius_scale),
                       neighbor_cid_count.array,
                       neighbor_cids.array,
                       start_indices,
                       neighbors)

    def _is_valid_nnps_wgs(self):
        # Max work group size can only be found by building the
        # kernel.
        try:
            find_neighbor_counts = self.helper.get_kernel(
                'find_neighbor_counts', sorted=self.sorted, wgs=self.leaf_size
            )

            find_neighbor = self.helper.get_kernel(
                'find_neighbors', sorted=self.sorted, wgs=self.leaf_size
            )
        except DeviceWGSException:
            return False
        else:
            return True

    def find_neighbor_lengths(self, neighbor_cid_count, neighbor_cids,
                              octree_src, neighbor_count,
                              use_partitions=False):
        self.check_nnps_compatibility(octree_src)

        n = self.pa.get_number_of_particles()
        wgs = self.leaf_size
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        def find_neighbor_counts_for_partition(partition_cids, partition_size,
                                               partition_wgs, q=None):
            find_neighbor_counts = self.helper.get_kernel(
                'find_neighbor_counts', sorted=self.sorted, wgs=wgs
            )
            find_neighbor_counts(partition_cids.array, octree_src.pids.array,
                                 self.pids.array,
                                 self.cids.array,
                                 octree_src.pbounds.array, self.pbounds.array,
                                 pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z,
                                 pa_gpu_src.h,
                                 pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z,
                                 pa_gpu_dst.h,
                                 dtype(self.radius_scale),
                                 neighbor_cid_count.array,
                                 neighbor_cids.array,
                                 neighbor_count,
                                 gs=(partition_wgs * partition_size,),
                                 ls=(partition_wgs,),
                                 queue=(get_queue() if q is None else q))

        if use_partitions and wgs > 32:
            if wgs < 128:
                wgs1 = 32
            else:
                wgs1 = 64

            m1, n1 = self.get_leaf_size_partitions(0, wgs1)

            find_neighbor_counts_for_partition(m1, n1, min(wgs, wgs1))
            m2, n2 = self.get_leaf_size_partitions(wgs1, wgs)
            find_neighbor_counts_for_partition(m2, n2, wgs)
        else:
            find_neighbor_counts_for_partition(
                self.unique_cids, self.unique_cid_count, wgs)

    def find_neighbors(self, neighbor_cid_count, neighbor_cids, octree_src,
                       start_indices, neighbors, use_partitions=False):
        self.check_nnps_compatibility(octree_src)

        wgs = self.leaf_size if self.leaf_size % 32 == 0 else \
            self.leaf_size + 32 - self.leaf_size % 32
        pa_gpu_dst = self.pa.gpu
        pa_gpu_src = octree_src.pa.gpu
        dtype = ctype_to_dtype(self.c_type)

        def find_neighbors_for_partition(partition_cids, partition_size,
                                         partition_wgs, q=None):
            find_neighbors = self.helper.get_kernel('find_neighbors',
                                                    sorted=self.sorted,
                                                    wgs=wgs)
            find_neighbors(partition_cids.array, octree_src.pids.array,
                           self.pids.array,
                           self.cids.array,
                           octree_src.pbounds.array, self.pbounds.array,
                           pa_gpu_src.x, pa_gpu_src.y, pa_gpu_src.z,
                           pa_gpu_src.h,
                           pa_gpu_dst.x, pa_gpu_dst.y, pa_gpu_dst.z,
                           pa_gpu_dst.h,
                           dtype(self.radius_scale),
                           neighbor_cid_count.array,
                           neighbor_cids.array,
                           start_indices,
                           neighbors,
                           gs=(partition_wgs * partition_size,),
                           ls=(partition_wgs,),
                           queue=(get_queue() if q is None else q))

        if use_partitions and wgs > 32:
            if wgs < 128:
                wgs1 = 32
            else:
                wgs1 = 64

            m1, n1 = self.get_leaf_size_partitions(0, wgs1)
            fraction = (n1 / int(self.unique_cid_count))

            if fraction > 0.3:
                find_neighbors_for_partition(m1, n1, wgs1)
                m2, n2 = self.get_leaf_size_partitions(wgs1, wgs)
                assert (n1 + n2 == self.unique_cid_count)
                find_neighbors_for_partition(m2, n2, wgs)
                return
        else:
            find_neighbors_for_partition(
                self.unique_cids, self.unique_cid_count, wgs)

    def check_nnps_compatibility(self, octree):
        """Check if octree types and parameters are compatible for NNPS

        Two octrees must satisfy a few conditions so that NNPS can be performed
        on one octree using the other as reference. In this case, the following
        conditions must be satisfied -

        1) Currently both should be instances of point_octree.OctreeGPUNNPS
        2) Both must have the same sortedness
        3) Both must use the same floating-point datatype
        4) Both must have the same leaf sizes

        Parameters
        ----------
        octree
        """
        # if not isinstance(octree, OctreeGPUNNPS):
        #     raise IncompatibleOctreesException(
        #         "Both octrees must be of the same type for NNPS"
        #     )

        if self.sorted != octree.sorted:
            raise IncompatibleOctreesException(
                "Octree sortedness need to be the same for NNPS"
            )

        if self.c_type != octree.c_type or \
                        self.use_double != octree.use_double:
            raise IncompatibleOctreesException(
                "Octree floating-point data types need to be the same for NNPS"
            )

        if self.leaf_size != octree.leaf_size:
            raise IncompatibleOctreesException(
                "Octree leaf sizes need to be the same for NNPS (%d != %d)" % (
                    self.leaf_size, octree.leaf_size)
            )

        return
