//CL//
<%! from itertools import product %>

<%def name="preamble(data_t)" cached="True">
    #define IN_BOUNDS(X, MIN, MAX) ((X >= MIN) && (X < MAX))
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))
    #define MIN(X, Y) ((X) < (Y) ? (X) : (Y))
    #define MAX(X, Y) ((X) > (Y) ? (X) : (Y))
    #define AVG(X, Y) (((X) + (Y)) / 2)
    #define ABS(X) ((X) > 0 ? (X) : -(X))
    #define EPS 1e-6
    #define INF 1e6
    #define SQR(X) ((X) * (X))

    char eye_index(ulong sfc, ulong mask, char rshift) {
        return ((sfc & mask) >> rshift);
    }

    char contains(private int *n1, private int *n2) {
        // Check if node n1 contains node n2
        char res = 1;
        %for i in range(3):
            res = res && (n1[${i}] <= n2[${i}]) && (n1[3 + ${i}] >= n2[3 + ${i}]);
        %endfor

        return res;
    }

    char contains_search(private ${data_t} *n1, private ${data_t} *n2) {
        // Check if node n1 contains node n2 with n1 having its search radius extension
        ${data_t} h = n1[6];
        char res = 1;
        %for i in range(3):
            res = res & (n1[${i}] - h <= n2[${i}]) & (n1[3 + ${i}] + h >= n2[3 + ${i}]);
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
            cdist = fabs((n1[${i}] + n1[3 + ${i}]) / 2 - (n2[${i}] + n2[3 + ${i}]) / 2);
            w1 = fabs(n1[${i}] - n1[3 + ${i}]);
            w2 = fabs(n2[${i}] - n2[3 + ${i}]);
            wavg = AVG(w1, w2);
            res &= (cdist - wavg <= h);
        % endfor

        return res;
    }

    char sphere_box_intersects(${data_t}4 s, private ${data_t} *n2) {
        /*
         * s = {x, y, z, r}
         */
        ${data_t} dist = 0;
        ${data_t} h = MAX(s.s3, n2[6]);
        %for i in range(3):
            if ((s.s${i} - n2[${i}]) * (s.s${i} - n2[3 + ${i}]) > 0) {
                dist += SQR(MIN(fabs(s.s${i} - n2[${i}]), fabs(s.s${i} - n2[${i} + 3])));
            }
        %endfor
        return dist < SQR(h);
    }
</%def>

<%def name="fill_particle_data_args(data_t)" cached="False">
    ${data_t}* x, ${data_t}* y, ${data_t}* z,
    ${data_t} cell_size, ${data_t}3 min,
    unsigned long* keys, unsigned int* pids
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
</%def>

<%def name="reorder_particles_args()" cached="False">
    int *pids,
    ulong *sfc, int *cids,
    char *seg_flag,
    uint2 *pbounds,
    int *offsets,
    uint8 *octant_vector,
    int *pids_next, ulong *sfc_next,
    int *cids_next,
    ulong mask, char rshift,
    uint csum_nodes_prev
</%def>

<%def name="reorder_particles_src()" cached="False">
    if (cids[i] < csum_nodes_prev || offsets[cids[i] - csum_nodes_prev] == -1) {
        sfc_next[i] = sfc[i];
        cids_next[i] = cids[i];
        pids_next[i] = pids[i];
    } else {
        uint2 pbound_here = pbounds[cids[i] - csum_nodes_prev];
        char octant = eye_index(sfc[i], mask, rshift);

        global uint *octv = (global uint *)(octant_vector + i);
        int sum = octv[octant];
        sum -= (octant == 0) ? 0 : octv[octant - 1];
        octv = (global uint *)(octant_vector + pbound_here.s1 - 1);
        sum += (octant == 0) ? 0 : octv[octant - 1];

        sfc_next[pbound_here.s0 + sum - 1] = sfc[i];
        pids_next[pbound_here.s0 + sum - 1] = pids[i];
        cids_next[pbound_here.s0 + sum - 1] = offsets[cids[i] - csum_nodes_prev] + octant;
    }
</%def>

<%def name="append_layer_args()" cached="False">
    int *offsets_next, uint2 *pbounds_next,
	int *offsets, uint2 *pbounds,
    int curr_offset, char is_last_level
</%def>

<%def name="append_layer_src()" cached="False">
	pbounds[curr_offset + i] = pbounds_next[i];
    offsets[curr_offset + i] = is_last_level ? -1 : offsets_next[i];
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

<%def name="set_node_bounds_args(data_t, sorted)" cached="False">
    uint2 *pbounds, int *offsets, int *pids,
    ${data_t} *gnode_data,
    ${data_t} *x0, ${data_t} *x1, ${data_t} *x2, ${data_t} *h,
    int csum_nodes
</%def>

<%def name="set_node_bounds_src(data_t, sorted)" cached="False">
    int cidx = i;
    % for i in range(3):
        ${data_t} xmin${i} = INF;
        ${data_t} xmax${i} = {-INF};
    % endfor

    ${data_t} hmax = 0;
    global ${data_t} *node_data;
    if (offsets[cidx] == -1) {
        // Leaf

        uint2 pbound_here = pbounds[cidx];
        int pid;
        for (int j=pbound_here.s0; j < pbound_here.s1; j++) {
            %if sorted:
                pid = j;
            % else:
                pid = pids[j];
            %endif
            % for i in range(3):
                xmin${i} = MIN(xmin${i}, x${i}[pid]);
                xmax${i} = MAX(xmax${i}, x${i}[pid]);
                hmax = MAX(hmax, h[pid]);
            % endfor

        }
    } else {
        // Non-leaf
        node_data = gnode_data + 8 * offsets[cidx];
        for (int j=0; j < 8; j++) {
            % for i in range(3):
                xmin${i} = MIN(xmin${i}, node_data[${i}]);
                xmax${i} = MAX(xmax${i}, node_data[3 + ${i}]);
            % endfor
            hmax = MAX(hmax, node_data[6]);
            node_data += 8;
        }
    }

    // Update current node
    node_data = gnode_data + 8 * cidx;
    % for i in range(3):
        node_data[${i}] = xmin${i};
        node_data[${i} + 3] = xmax${i};
    % endfor
    node_data[6] = hmax;
    node_data[7] = offsets[cidx];
</%def>

<%def name="dfs_template(data_t)" cached="False">

    ${data_t} ndst[8];
    ${data_t} nsrc[8];

    int cid = cids[unique_cid_idx[i]];
    % for i in range(8):
        ndst[${i}] = n1[cid * 8 + ${i}];
    % endfor

    /*
	 * Assuming max depth of 16
	 * stack_idx is also equal to current layer of octree
	 * child_idx = number of children iterated through
	 * idx_stack = current node
	 */
	char child_stack[16];
	int cid_stack[16];
	char idx = 0;
	child_stack[0] = 0;
	cid_stack[0] = 1;
    char flag;
    int curr_cid;

    ${caller.pre_loop()}
	while (idx >= 0) {

        // Recurse to find either leaf node or invalid node
		curr_cid = cid_stack[idx];
        flag = 1;
        % for i in range(8):
            nsrc[${i}] = n2[8 * curr_cid + ${i}];
        % endfor
        while (offsets[curr_cid] != -1) {

            if (!intersects(ndst, nsrc) && !contains(nsrc, ndst)) {
                flag = 0;
                break;
            }

            idx++;
            curr_cid = offsets[curr_cid];
            cid_stack[idx] = curr_cid;
            child_stack[idx] = 0;
            % for i in range(8):
                nsrc[${i}] = n2[8 * curr_cid + ${i}];
            % endfor
        }

        // Process
        if (intersects(ndst, nsrc) || contains_search(ndst, nsrc)) {
            ${caller.query()}
        }

        // Recurse back to find node with a valid neighbor
        while (child_stack[idx] >= 7 && idx >= 0)
            idx--;

        // Iterate to next neighbor
        if (idx >= 0) {
            cid_stack[idx]++;
            child_stack[idx]++;
        }
    }

    ${caller.post_loop()}
</%def>

<%def name="find_neighbor_cid_counts_args(data_t)" cached="False">\
    int *unique_cid_idx, int *cids, uint2 *pbounds, int *offsets, ${data_t} *n1, ${data_t} *n2, int *cnt
</%def>

<%def name="find_neighbor_cid_counts_src(data_t)" cached="False">\
    <%self:dfs_template data_t="${data_t}">
        <%def name="pre_loop()">
            int count = 0;
        </%def>
        <%def name="query()">
            if (pbounds[curr_cid].s0 < pbounds[curr_cid].s1)
                count++;
            /*if (i == 100 && pbounds[curr_cid].s0 < pbounds[curr_cid].s1) {
            printf("%d %d\n", idx, curr_cid);
            printf("%d %d\n", pbounds[curr_cid].s0, pbounds[curr_cid].s1);
            printf("%d %d\n", intersects(ndst, nsrc), contains_search(ndst, nsrc));
            printf("%f %f %f\t%f %f %f\n", ndst[0], ndst[1], ndst[2], ndst[3], ndst[4], ndst[5], ndst[6]);
            printf("%f %f %f\t%f %f %f\n", nsrc[0], nsrc[1], nsrc[2], nsrc[3], nsrc[4], nsrc[5], nsrc[6]);
            printf("\n");
            }*/
        </%def>
        <%def name="post_loop()">
            cnt[i] = count;
        </%def>
    </%self:dfs_template>
</%def>

<%def name="find_neighbor_cids_args(data_t)" cached="False">\
    int *unique_cid_idx, int *cids, uint2 *pbounds, int *offsets, ${data_t} *n1, ${data_t} *n2, int *cnt,
    int *neighbor_cids
</%def>

<%def name="find_neighbor_cids_src(data_t)" cached="False">\
    <%self:dfs_template data_t="${data_t}">
        <%def name="pre_loop()">
            int offset = cnt[i];
        </%def>
        <%def name="query()">
            if (pbounds[curr_cid].s0 < pbounds[curr_cid].s1)
                neighbor_cids[offset++] = curr_cid;
        </%def>
        <%def name="post_loop()">
        </%def>
    </%self:dfs_template>
</%def>

<%def name="find_neighbors_template(data_t, sorted, wgs)" cached="False">
    int idx = i / ${wgs};

    // Fetch src particles
    ${data_t} xd, yd, zd, hd;

    int cid_dst = cids[unique_cid_idx[idx]];
    uint2 pbound_here = pbounds[cid_dst];
    char svalid = (pbound_here.s0 + lid < pbound_here.s1);
    int pid;

    if (svalid) {
        % if sorted:
            pid = pbound_here.s0 + lid;
        % else:
            pid = pids[pbound_here.s0 + lid];
        % endif

        xd = x[pid];
        yd = y[pid];
        zd = z[pid];
        hd = h[pid];
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
    ${data_t} r2;


    ${caller.pre_loop()}
    while (offset_src < offset_lim) {
        cid_src = neighbor_cids[offset_src];
        pbound_here2 = pbounds[cid_src];
        offset_src++;

        // Copy src data
        if (pbound_here2.s0 + lid < pbound_here2.s1) {
            %if sorted:
                pid_src = pbound_here2.s0 + lid;
            % else:
                pid_src = pids[pbound_here2.s0 + lid];
            %endif
            xs[lid] = x[pid_src];
            ys[lid] = y[pid_src];
            zs[lid] = z[pid_src];
            hs[lid] = h[pid_src];
        }
        m = pbound_here2.s1 - pbound_here2.s0;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (svalid) {
            for (int j=0; j < m; j++) {
                % if sorted:
                    pid_src= j;
                % else:
                    pid_src = pids[j];
                % endif
                ${data_t} dist2 = NORM2(xs[j] - xd,
                                        ys[j] - yd,
                                        zs[j] - zd);

                r2 = MAX(hs[j], hd);
                r2 *= r2;
                if (dist2 < r2) {
                    ${caller.query()}
                }
            }
        }
    }
    ${caller.post_loop()}
</%def>

<%def name="find_neighbor_counts_args(data_t, sorted, wgs)" cached="False">
    int *unique_cid_idx, int *pids, int *cids,
    uint2 *pbounds, int *offsets,
    ${data_t} *x, ${data_t} *y, ${data_t} *z, ${data_t} *h,
    int *neighbor_cid_offset, int *neighbor_cids,
    int *neighbor_counts
</%def>
<%def name="find_neighbor_counts_src(data_t, sorted, wgs)" cached="False">
     <%self:find_neighbors_template data_t="${data_t}" sorted="${sorted}" wgs="${wgs}">
        <%def name="pre_loop()">
            int count = 0;
        </%def>
        <%def name="query()">
            count++;
        </%def>
        <%def name="post_loop()">
            if(svalid)
                neighbor_counts[pid] = count;
        </%def>
    </%self:find_neighbors_template>
</%def>

<%def name="find_neighbors_args(data_t, sorted, wgs)" cached="False">
</%def>
<%def name="find_neighbors_src(data_t, sorted, wgs)" cached="False">
     <%self:find_neighbors_template data_t="${data_t}" sorted="${sorted}" wgs="${wgs}">
        <%def name="pre_loop()">
            int offset = neighbor_counts[pid];
        </%def>
        <%def name="query()">
            neighbors[offset++] = offset_lim;
        </%def>
        <%def name="post_loop()">
        </%def>
    </%self:find_neighbors_template>
</%def>