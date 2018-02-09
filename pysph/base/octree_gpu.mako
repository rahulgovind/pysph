//CL//
<%! from itertools import product %>

<%def name="preamble()" cached="True">
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))

    inline uint find_smallest_containing_node(global uint *offsets, ulong key, char max_depth_here, char max_depth)
    {
        uint curr = 0;
	    char depth = 0;
	    char rshift;

	    while (depth < max_depth_here) {
		    if (offset[curr] == -1)
			    break;

		    rshift = 3 * (max_depth - depth);
		    curr = offset[curr] + ((key & (7 << rshift)) >> rshift);;
		    depth++;
	    }

	    return curr;
    }
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
    radius[i] = float(radius_scale * h[i]);
    levels[i] = max_depth - (char)last_set_bit_pos(floor(radius[i] / cell_size));
</%def>


<%def name="append_layer_args()" cached="True">
	int *offsets, uint2 *pbounds, int *offsets_next, uint2 *pbounds_next,
    int curr_offset, int next_offset
</%def>
	    
<%def name="append_layer_src()" cached="True">
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
</%def>

<%def name="store_neighbour_cids_args(data_t, sorted)" cached="True">
    int *offsets, uint2 *pbounds,
    ${data_t} *x, ${data_t} *y, ${data_t} *z, ${data_t} cell_size,
    int *pids, int *neighbour_cids,
    char octree_depth, char max_depth
</%def>

<%def name="store_neighbour_cids_src(data_t, sorted)" cached="True">
    int depth_here = 0;
    while (depth_here <= octree_depth) {
        if (i < csum_nodes[depth_here])
            break;
        depth_here++;
    }

    int pid = pbounds[i].s1 - 1;

    % if not sorted:
        pid = pids[pid];
    % endif

    uint3 c = uint3(floor((x[pid] - min.x) / cell_size),
                    floor((y[pid] - min.y) / cell_size),
                    floor((z[pid] - min.z) / cell_size));
    c.x >>= (max_depth - depth_here);
    c.y >>= (max_depth - depth_here);
    c.z >>= (max_depth - depth_here);
    int max_int = (1 << depth_here);

    int num_keys = 0;
    ulong keys[27];

    % for (i, j, k) in product(range(-1, 2), 3):
        if ((IN_BOUNDS(c+x + ${i}, 0, max_int) && IN_BOUNDS(c_y + ${j}, 0, max_int) && IN_BOUNDS(c_z + ${k}))) {
            keys[num_keys++] = (interleave(c_x + i, c_y + j, c_z + k) << (max_depth - depth_here));
        }
    % endfor

    insertion_sort(keys, num_keys);

    int unique_keys = 1;
    for (int j = 1; j < num_keys; j++) {
        if (keys[j] != keys[j-1])
            keys[unique_keys++] = keys[j];
    }

    for (int j = 0; j < 27; j++) {
        neighbour_cids[27 * i + j] = (j < unique_keys) ? find_smallest_containing_node(offsets, keys[j], depth_here + 1, max_depth) : -1;
    }
</%def>