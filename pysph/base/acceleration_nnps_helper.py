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
    int _idx = get_group_id(0);
    int cnt = 0;

    // Fetch dst particles
    ${data_t} _xd, _yd, _zd, _hd;

    int _cid_dst = _unique_cids[_idx];
    uint2 _pbound_here = _pbounds_dst[_cid_dst];
    char _svalid = (_pbound_here.s0 + lid < _pbound_here.s1);
    unsigned int d_idx;

    if (_svalid) {
        % if sorted:
        d_idx = _pbound_here.s0 + lid;
        % else:
        d_idx = _pids_dst[_pbound_here.s0 + lid];
        % endif

        _xd = _xdst[d_idx];
        _yd = _ydst[d_idx];
        _zd = _zdst[d_idx];
        _hd = _hdst[d_idx];
    }

    // Set loop parameters
    int _cid_src, _pid_src;
    int _offset_src = _neighbor_cid_offset[_idx];
    int _offset_lim = _neighbor_cid_offset[_idx + 1];
    uint2 _pbound_here2;
    int _m;
    local ${data_t} _xs[${wgs}];
    local ${data_t} _ys[${wgs}];
    local ${data_t} _zs[${wgs}];
    local ${data_t} _hs[${wgs}];
    % for var, type in zip(vars, types):
    local ${type} ${var}[${wgs}];
    % endfor
    ${data_t} _r2;

    ${setup}
    while (_offset_src < _offset_lim) {
        _cid_src = _neighbor_cids[_offset_src];
        _pbound_here2 = _pbounds_src[_cid_src];

        while (_pbound_here2.s0 < _pbound_here2.s1) {
            // Copy src data
            if (_pbound_here2.s0 + lid < _pbound_here2.s1) {

                %if sorted:
                _pid_src = _pbound_here2.s0 + lid;
                % else:
                _pid_src = _pids_src[_pbound_here2.s0 + lid];
                %endif

                _xs[lid] = _xsrc[_pid_src];
                _ys[lid] = _ysrc[_pid_src];
                _zs[lid] = _zsrc[_pid_src];
                _hs[lid] = _hsrc[_pid_src];

                % for var in vars:
                ${var}[lid] = ${var}_global[_pid_src];
                % endfor
            }
            _m = min(_pbound_here2.s1, _pbound_here2.s0 + ${wgs}) - _pbound_here2.s0;

            barrier(CLK_LOCAL_MEM_FENCE);

            // Everything this point forward is done independently
            // by each thread.
            if (_svalid) {
                for (int _j=0; _j < _m; _j++) {
                    int s_idx = _j;
                    ${data_t} _dist2 = NORM2(_xs[_j] - _xd,
                                            _ys[_j] - _yd,
                                            _zs[_j] - _zd);

                    _r2 = MAX(_hs[_j], _hd) * _radius_scale;
                    _r2 *= _r2;
                    if (_dist2 < _r2) {
                        ${loop_code}
                    }
                }
            }
            _pbound_here2.s0 += ${wgs};
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        _offset_src++;
    }

"""

NNPS_ARGS_TEMPLATE = """
    __global int *_unique_cids, __global int *_pids_src, __global int
    *_pids_dst,
    __global int *_cids,
    __global uint2 *_pbounds_src, __global uint2 *_pbounds_dst,
    __global %(data_t)s *_xsrc, __global %(data_t)s *_ysrc,
    __global %(data_t)s *_zsrc, __global %(data_t)s *_hsrc,
    __global %(data_t)s *_xdst, __global %(data_t)s *_ydst,
    __global %(data_t)s *_zdst, __global %(data_t)s *_hdst,
    %(data_t)s _radius_scale,
    __global int *_neighbor_cid_offset, __global int *_neighbor_cids
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


def generate_body(setup, loop, vars, types, wgs):
    return _generate_nnps_code(True, wgs, setup, loop, vars, types,
                               'float')


def get_kernel_args_list(data_t='float'):
    args = NNPS_ARGS_TEMPLATE % {'data_t': data_t}
    return args.split(",")
