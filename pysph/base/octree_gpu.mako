//CL//
<%! from itertools import product %>

<%def name="preamble()" cached="True">
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
    #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
    #define EPS 1e-6
    inline uint find_smallest_containing_node(global int *offsets, ulong key, char max_depth)
    {
        uint curr = 0;
	    char depth = 0;
	    char rshift;

	    while (depth < max_depth) {
		    if (offsets[curr] == -1)
			    break;

		    rshift = 3 * (max_depth - depth - 1);
		    curr = offsets[curr] + ((key & (7 << rshift)) >> rshift);
		    depth++;
	    }

	    return curr;
    }

    char eye_index(ulong sfc, ulong mask, char rshift, bool same_level) {
        return (same_level ? 8 : ((sfc & mask) >> rshift));
    }
</%def>

<%def name="post_build_args()" cached="False">
    char *levels, char octree_depth
</%def>

<%def name="post_build_src()" cached="False">
    levels[i] = MIN(levels[i], octree_depth);
</%def>

<%def name="fill_particle_data_args(data_t)" cached="False">
    ${data_t}* x, ${data_t}* y, ${data_t}* z, ${data_t}* h,
    ${data_t} cell_size, ${data_t}3 min,
    ${data_t} *radius, ${data_t} radius_scale,
    unsigned long* keys, unsigned int* pids,
    char *levels, char max_depth
</%def>

<%def name="fill_particle_data_src(data_t)" cached="False">
    unsigned long c_x, c_y, c_z;
    FIND_CELL_ID(
        x[i] - min.x,
        y[i] - min.y,
        z[i] - min.z,
        cell_size, c_x, c_y, c_z
        );
    unsigned long key;
    key = interleave(c_x, c_y, c_z);
    keys[i] = key;
    pids[i] = i;
    radius[i] = radius_scale * h[i];
    levels[i] = max_depth - (char)first_set_bit_pos((int)(radius[i] / cell_size));
</%def>

<%def name="reorder_particles_args()" cached="False">
    int *pids,
    ulong *sfc, int *cids,
    char *levels, char *seg_flag,
    uint2 *pbounds,
    int *offsets,
    uint8 *octant_vector,
    int *pids_next, ulong *sfc_next,
    int *cids_next,
    char *levels_next,
    char curr_level, ulong mask, char rshift,
    uint csum_nodes_prev
</%def>

<%def name="reorder_particles_src()" cached="False">
    if (cids[i] < csum_nodes_prev || offsets[cids[i] - csum_nodes_prev] == -1) {
        sfc_next[i] = sfc[i];
        levels_next[i] = levels[i];
        cids_next[i] = cids[i];
        pids_next[i] = pids[i];
    } else {
        uint2 pbound_here = pbounds[cids[i] - csum_nodes_prev];
        char octant = eye_index(sfc[i], mask, rshift, levels[i] == curr_level);

        global uint *octv = (global uint *)(octant_vector + i);
        int sum = (octant == 8) ? (i - pbound_here.s0  + 1) : octv[octant];
        sum -= (octant == 0) ? 0 : octv[octant - 1];
        octv = (global uint *)(octant_vector + pbound_here.s1 - 1);
        sum += (octant == 0) ? 0 : octv[octant - 1];

        levels_next[pbound_here.s0 + sum - 1] = levels[i];
        sfc_next[pbound_here.s0 + sum - 1] = sfc[i];
        pids_next[pbound_here.s0 + sum - 1] = pids[i];
        cids_next[pbound_here.s0 + sum - 1] = (octant == 8 ? cids[i] : (offsets[cids[i] - csum_nodes_prev] + octant));
    }
</%def>

<%def name="append_layer_args()" cached="False">
    int *offsets_next, uint2 *pbounds_next,
	int *offsets, uint2 *pbounds,
    int curr_offset
</%def>

<%def name="append_layer_src()" cached="False">
	pbounds[curr_offset + i] = pbounds_next[i];
    offsets[curr_offset + i] = offsets_next[i];
</%def>

<%def name="set_node_data_args()", cached="True">
    int *offsets_prev, uint2 *pbounds_prev,
    int *offsets, uint2 *pbounds,
    char *seg_flag, uint8 *octant_vector,
    uint csum_nodes
</%def>

<%def name="set_node_data_src()", cached="True">
    uint2 pbound_here = pbounds_prev[i];
    int child_offset = offsets_prev[i];
    if (child_offset == -1) {
        PYOPENCL_ELWISE_CONTINUE;
    }
    child_offset -= csum_nodes;

    if (pbound_here.s1 > pbound_here.s0) {
        global uint *octv = (global uint *)(octant_vector + pbound_here.s1 - 1);
        % for i in range(8):
            % if i == 0:
                pbounds[child_offset] = (uint2)(pbound_here.s0,
                                                pbound_here.s0 + octv[0]);
                seg_flag[pbound_here.s0] = 1;
            % else:
                pbounds[child_offset + ${i}] = (uint2)(pbound_here.s0 + octv[${i - 1}],
                                                       pbound_here.s0 + octv[${i}]);
                seg_flag[pbound_here.s0 + octv[${i - 1}]] = 1;
            % endif
        % endfor
    } else {
        % for i in range(8):
            pbounds[child_offset + ${i}] = pbound_here;
        % endfor
    }
</%def>

<%def name="store_neighbor_cids_args(data_t, sorted)" cached="False">
    int *unique_cids_idx,
    int *pids,
    int *offsets_dst, uint2 *pbounds_dst,
    ${data_t} *x, ${data_t} *y, ${data_t} *z, ${data_t} cell_size,
    ${data_t}3 min,
    int *neighbour_cids,
    char *levels,
    char octree_depth_dst, char max_depth
</%def>

<%def name="store_neighbor_cids_src(data_t, sorted)" cached="False">
    int idx = unique_cids_idx[i];
    char depth_here = MIN(levels[idx] - 1, octree_depth_dst);
    int pid;
    % if not sorted:
        pid = pids[idx];
    % else:
        pid = idx;
    % endif
    uint c_x = floor((x[pid] - min.x) / cell_size);
    uint c_y = floor((y[pid] - min.y) / cell_size);
    uint c_z = floor((z[pid] - min.z) / cell_size);

    uint3 c = uint3(floor((x[pid] - min.x) / cell_size),
                    floor((y[pid] - min.y) / cell_size),
                    floor((z[pid] - min.z) / cell_size));
    c_x >>= (max_depth - depth_here - 1);
    c_y >>= (max_depth - depth_here - 1);
    c_z >>= (max_depth - depth_here - 1);
    int max_int = (1 << depth_here);
    ulong key_orig = interleave(c.x, c.y , c.z);
    int num_keys = 0;
    ulong keys[27];

    % for i, j, k in product(range(-1, 2), repeat=3):
        if ((IN_BOUNDS(c_x + (${i}), 0, max_int) && IN_BOUNDS(c_y + (${j}), 0, max_int) && IN_BOUNDS(c_z + (${k}), 0, max_int))) {
            ulong key = interleave(c_x + (${i}), c_y + (${j}), c_z + (${k}));
            keys[num_keys++] = find_smallest_containing_node(offsets_dst, key, depth_here);
        }
    % endfor

    insertion_sort(keys, num_keys);

    int unique_keys = 1;
    for (int j = 1; j < num_keys; j++) {
        if (keys[j] != keys[j-1])
            keys[unique_keys++] = keys[j];
    }

    for (int j = 0; j < 27; j++) {
        neighbour_cids[i * 27 + j] = (j < unique_keys) ? keys[j] : -1;
    }

</%def>

<%def name="sort_particles_args(data_t, args1, args2)" cached="False">
    <%
        args = ','.join([data_t + ' *' + x for x in (args1 + args2)])
    %>
    int *pids, ${args}
</%def>

<%def name="sort_particles_src(data_t, args1, args2)" cached="False">
    %for arg1, arg2 in zip(args1, args2):
        ${arg2}[i] = ${arg1}[pids[i]];
    %endfor
</%def>

<%def name="copy_back_args(data_t, args1, args2)" cached="False">
    <%
        args = ','.join([data_t + ' *' + x for x in (args1 + args2)])
    %>
    int *pids, ${args}
</%def>

<%def name="copy_back_src(data_t, args1, args2)" cached="False">
    %for arg1, arg2 in zip(args1, args2):
        ${arg1}[i] = ${arg2}[i];
    %endfor
</%def>

<%def name="copy_int_args()" cached="False">\
    int *src, int *dst
</%def>

<%def name="copy_int_src()" cached="False">\
    dst[i] = src[i];
</%def>
<%def name="store_neighbor_counts_args(data_t, sorted)" cached="False">
    int *pids_src, int *pids_dst,
    int *unique_cids_map,
    uint2 *pbounds_dst,
    char *levels_src, char *levels_dst,
    int *neighbor_cids,
    ${data_t} *x_src, ${data_t} *y_src, ${data_t} *z_src, ${data_t} *h_src,
    ${data_t} *x_dst, ${data_t} *y_dst, ${data_t} *z_dst, ${data_t} *h_dst,
    int *neighbor_counts_src, int *neighbor_counts_dst,
    ${data_t} radius_scale
</%def>

<%def name="store_neighbor_counts_src(data_t, sorted)" cached="False">
    int pid_src, pbound_idx, pid_dst;
    % if not sorted:
        pid_src = pids_src[i];
    % else:
        pid_src = i;
    % endif

    int idx = 27 * unique_cids_map[i];
    ${data_t} r_src2 = h_src[pid_src] * radius_scale;
    r_src2 *= r_src2;
    ${data_t} r_dst2;
    ${data_t} xs = x_src[pid_src];
    ${data_t} ys = y_src[pid_src];
    ${data_t} zs = z_src[pid_src];
    char ls = levels_src[i];

    int large_nbr_count = 0;
    for (int k = 0; k < 27; k++) {
        if (neighbor_cids[idx + k] < 0)
            break;

        pbound_idx = neighbor_cids[idx + k];
        uint2 pbound_here = pbounds_dst[pbound_idx];
        for (int j = pbound_here.s0; j < pbound_here.s1; j++) {
            % if not sorted:
                pid_dst = pids_dst[j];
            % else:
                pid_dst = j;
            % endif

            ${data_t} dist2 = NORM2(xs - x_dst[pid_dst],
                                    ys - y_dst[pid_dst],
                                    zs - z_dst[pid_dst]);



            if (dist2 <= r_src2 && ls <= levels_dst[j]) {
                large_nbr_count++;
                r_dst2 = h_dst[pid_dst] * radius_scale;
                r_dst2 *= r_dst2;
                if (dist2 > r_dst2 || ls < levels_dst[j]) {
                        atom_inc(neighbor_counts_dst + pid_dst);
                }
            }
        }
    }
    atom_add(neighbor_counts_src + pid_src, large_nbr_count);
</%def>


<%def name="store_neighbors_args(data_t, sorted)" cached="False">
    int *pids_src, int *pids_dst,
    int *unique_cids_map,
    uint2 *pbounds_dst,
    char *levels_src, char *levels_dst,
    int *neighbor_cids,
    ${data_t} *x_src, ${data_t} *y_src, ${data_t} *z_src, ${data_t} *h_src,
    ${data_t} *x_dst, ${data_t} *y_dst, ${data_t} *z_dst, ${data_t} *h_dst,
    int *neighbor_counts_src, int *neighbor_counts_dst,
    int *neighbors_src, int *neighbors_dst,
    ${data_t} radius_scale
</%def>

<%def name="store_neighbors_src(data_t, sorted)" cached="False">
    <%
        buffer1 = 4
    %>

    int pid_src = i;
    % if not sorted:
        pid_src = pids_src[pid_src];
    % endif

    int idx = 27 * unique_cids_map[i];
    int pbound_idx, pid_dst;
    ${data_t} r_src2 = h_src[pid_src] * radius_scale;
    r_src2 *= r_src2;
    ${data_t} r_dst2;
    ${data_t} xs = x_src[pid_src];
    ${data_t} ys = y_src[pid_src];
    ${data_t} zs = z_src[pid_src];
    char ls = levels_src[i];

    int poffset = neighbor_counts_src[pid_src];
    int large_nbrs[${buffer1}];
    char large_nbr_count =  0;

    for (int k = 0; k < 27; k++) {
        if (neighbor_cids[idx + k] < 0)
            break;

        pbound_idx = neighbor_cids[idx + k];
        uint2 pbound_here = pbounds_dst[pbound_idx];

        for (int j = pbound_here.s0; j < pbound_here.s1; j++) {
            % if not sorted:
                pid_dst = pids_dst[j];
            % else:
                pid_dst = j;
            % endif

            ${data_t} dist2 = NORM2(xs - x_dst[pid_dst],
                                    ys - y_dst[pid_dst],
                                    zs - z_dst[pid_dst]);

            if (dist2 <= r_src2 && ls <= levels_dst[j]) {
                large_nbrs[large_nbr_count++] = pid_dst;
                if (large_nbr_count == ${buffer1}) {
                    int m = atom_add(neighbor_counts_src + pid_src, large_nbr_count);
                    for (int m2=0; m2 < ${buffer1}; m2++) {
                        neighbors_src[m+m2] = large_nbrs[m2];
                    }
                    large_nbr_count = 0;
                }
                r_dst2 = h_dst[pid_dst] * radius_scale;
                r_dst2 *= r_dst2;
                if (dist2 > r_dst2 || ls < levels_dst[j]) {
                    neighbors_dst[atom_inc(neighbor_counts_dst + pid_dst)] = pid_src;
                }
            }
        }
    }

    int m = atom_add(neighbor_counts_src + pid_src, large_nbr_count);

    for (int m2=0; m2 < large_nbr_count; m2++) {
        neighbors_src[m+m2] = large_nbrs[m2];
    }

</%def>