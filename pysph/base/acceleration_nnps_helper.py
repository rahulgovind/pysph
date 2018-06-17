import sys
from mako.template import Template

disable_unicode = False if sys.version_info.major > 2 else True

NNPS_TEMPLATE = r"""

    /*
     * Property owners
     * octree_dst: cids, unique_cids
     * octree_src: neighbor_cid_offset, neighbor_cids
     * self evident: xsrc, ysrc, zsrc, hsrc,
                     xdst, ydst, zdst, hdst,
                     pbounds_src, pbounds_dst,
     */
    long i = get_global_id(0);
    int lid = get_local_id(0);
    int idx = i / ${wgs};
    int cnt = 0;

    // Fetch dst particles
    ${data_t} xd, yd, zd, hd;

    int cid_dst = unique_cids[idx];
    uint2 pbound_here = pbounds_dst[cid_dst];
    char svalid = (pbound_here.s0 + lid < pbound_here.s1);
    unsigned int d_idx;

    if (svalid) {
        % if sorted:
            d_idx = pbound_here.s0 + lid;
        % else:
            d_idx = pids_dst[pbound_here.s0 + lid];
        % endif

        xd = xdst[d_idx];
        yd = ydst[d_idx];
        zd = zdst[d_idx];
        hd = hdst[d_idx];
    }

    // Set loop parameters
    int cid_src, pid_src;
    int offset_src = neighbor_cid_offset[idx];
    int offset_lim = neighbor_cid_offset[idx + 1];
    uint2 pbound_here2;
    int m;
    local ${data_t} xs[${wgs}];
    local ${data_t} ys[${wgs}];
    local ${data_t} zs[${wgs}];
    local ${data_t} hs[${wgs}];
    % for var, type in zip(vars, types):
        // local ${type} ${var}[${wgs}];
        __global ${type}* ${var} = ${var}_global;
    % endfor
    ${data_t} r2;

    __global int *_neighbors = neighbors + start_indices[d_idx];
    ${setup}
    while (offset_src < offset_lim) {
        cid_src = neighbor_cids[offset_src];
        pbound_here2 = pbounds_src[cid_src];

        offset_src++;

        while (pbound_here2.s0 < pbound_here2.s1) {
            // Copy src data
            if (pbound_here2.s0 + lid < pbound_here2.s1) {
                %if sorted:
                    pid_src = pbound_here2.s0 + lid;
                % else:
                    pid_src = pids_src[pbound_here2.s0 + lid];
                %endif
                xs[lid] = xsrc[pid_src];
                ys[lid] = ysrc[pid_src];
                zs[lid] = zsrc[pid_src];
                hs[lid] = hsrc[pid_src];

                % for var in vars:
                    // ${var}[lid] = ${var}_global[pid_src];
                % endfor
            }
            m = min(pbound_here2.s1, pbound_here2.s0 + ${wgs}) - pbound_here2.s0;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Everything this point forward is done independently
            // by each thread.
            if (svalid) {
                for (int j=0; j < m; j++) {
                    %if sorted:
                        int s_idx = pbound_here2.s0 + j;
                    %else:
                        int s_idx = pids_src[pbound_here2.s0 + j];
                    %endif
                    ${data_t} dist2 = NORM2(xs[j] - xd,
                                            ys[j] - yd,
                                            zs[j] - zd);

                    r2 = MAX(hs[j], hd) * radius_scale;
                    r2 *= r2;
                    if (dist2 < r2) {
                        _neighbors[cnt++] = s_idx;
                        ${loop_code}
                    }
                }
            }
            pbound_here2.s0 += ${wgs};
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    if (svalid) {
        neighbor_counts[d_idx] = cnt;
    }
"""

NNPS_ARGS_TEMPLATE = """
    __global int *unique_cids, __global int *pids_src, __global *pids_dst,
    __global int *cids,
    __global uint2 *pbounds_src, __global uint2 *pbounds_dst,
    __global %(data_t)s *xsrc, __global %(data_t)s *ysrc,
    __global %(data_t)s *zsrc, __global %(data_t)s *hsrc,
    __global %(data_t)s *xdst, __global %(data_t)s *ydst,
    __global %(data_t)s *zdst, __global %(data_t)s *hdst,
    %(data_t)s radius_scale,
    __global int *neighbor_cid_offset, __global int *neighbor_cids,
    __global int *start_indices,
    __global int *neighbor_counts, __global int *neighbors
    """


def _generate_nnps_code(sorted, wgs, setup, loop, vars, types,
                        data_t='float'):
    # Note: Properties like the data type and sortedness
    # need to be fixed throughout the simulation since
    # currently this function is only called at the start of
    # the simulation.
    return Template(NNPS_TEMPLATE, disable_unicode=disable_unicode).render(
        data_t=data_t, sorted=sorted, wgs=wgs, setup=setup, loop_code=loop,
        vars=vars, types=types
    )


def generate_body(setup, loop, vars, types):
    return _generate_nnps_code(True, 32, setup, loop, vars, types,
                               'float')


def get_kernel_args_list(data_t='float'):
    args = NNPS_ARGS_TEMPLATE % {'data_t': data_t}
    return args.split(",")
