//CL//

<%def name="get_helpers()" cached="False">
    #define NORM2(X, Y, Z) ((X)*(X) + (Y)*(Y) + (Z)*(Z))

    #define FIND_CELL_ID(x, y, z, h, c_x, c_y, c_z) \
        c_x = floor((x)/h); c_y = floor((y)/h); c_z = floor((z)/h)

    inline ulong interleave(ulong p, \
            ulong q, ulong r);

    inline int neighbor_boxes(int c_x, int c_y, int c_z, \
            ulong* nbr_boxes);

    inline ulong interleave(ulong p, \
            ulong q, ulong r)
    {
        p = (p | (p << 32)) & 0x1f00000000ffff;
        p = (p | (p << 16)) & 0x1f0000ff0000ff;
        p = (p | (p <<  8)) & 0x100f00f00f00f00f;
        p = (p | (p <<  4)) & 0x10c30c30c30c30c3;
        p = (p | (p <<  2)) & 0x1249249249249249;

        q = (q | (q << 32)) & 0x1f00000000ffff;
        q = (q | (q << 16)) & 0x1f0000ff0000ff;
        q = (q | (q <<  8)) & 0x100f00f00f00f00f;
        q = (q | (q <<  4)) & 0x10c30c30c30c30c3;
        q = (q | (q <<  2)) & 0x1249249249249249;

        r = (r | (r << 32)) & 0x1f00000000ffff;
        r = (r | (r << 16)) & 0x1f0000ff0000ff;
        r = (r | (r <<  8)) & 0x100f00f00f00f00f;
        r = (r | (r <<  4)) & 0x10c30c30c30c30c3;
        r = (r | (r <<  2)) & 0x1249249249249249;

        return (p | (q << 1) | (r << 2));
    }

    inline int find_idx(__global ulong* keys, \
            int num_particles, ulong key)
    {
        int first = 0;
        int last = num_particles - 1;
        int middle = (first + last) / 2;

        while(first <= last)
        {
            if(keys[middle] < key)
                first = middle + 1;
            else if(keys[middle] > key)
                last = middle - 1;
            else if(keys[middle] == key)
            {
                if(middle == 0)
                    return 0;
                if(keys[middle - 1] != key)
                    return middle;
                else
                    last = middle - 1;
            }
            middle = (first + last) / 2;
        }

        return -1;
    }

    inline int neighbor_boxes(int c_x, int c_y, int c_z, \
        ulong* nbr_boxes)
    {
        int nbr_boxes_length = 1;
        int j, k, m;
        ulong key;
        nbr_boxes[0] = interleave(c_x, c_y, c_z);

        #pragma unroll
        for(j=-1; j<2; j++)
        {
            #pragma unroll
            for(k=-1; k<2; k++)
            {
                #pragma unroll
                for(m=-1; m<2; m++)
                {
                    if((j != 0 || k != 0 || m != 0) && c_x+m >= 0 && c_y+k >= 0 && c_z+j >= 0)
                    {
                        key = interleave(c_x+m, c_y+k, c_z+j);
                        nbr_boxes[nbr_boxes_length] = key;
                        nbr_boxes_length++;
                    }
                }
            }
        }

        return nbr_boxes_length;
    }

    inline char first_set_bit_pos(int x)
    {
        char result = 0;
        // x = x ^ (x & (x - 1));
        while (x > 0) {
            x >>= 1;
            result++;
        }

        return result;
    }

    inline void insertion_sort(unsigned long* arr, int n)
    {
        int i, j;
        unsigned long key;
        for (i = 1; i < n; i++)
        {
            key = arr[i];
            j = i-1;

            while (j >= 0 && arr[j] > key)
            {
                arr[j+1] = arr[j];
                j = j-1;
            }
            arr[j+1] = key;
        }
    }


</%def>


