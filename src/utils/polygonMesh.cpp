#include "polygonMesh.h"

namespace polygonMesh
{
    vector<size_t> edge2vtx_from_polygon_mesh(
        const vector<size_t> &elem2idx,
        const vector<size_t> &idx2vtx,
        size_t num_vtx)
    {
        // ---- 1. 顶点 → 元素：构建 vtx2elem ----
        auto vtx2elem_pair = vtx2elem_from_polygon_mesh(elem2idx, idx2vtx, num_vtx);
        const vector<size_t> &vtx2jdx = vtx2elem_pair.first;
        const vector<size_t> &jdx2elem = vtx2elem_pair.second;

        // ---- 2. 顶点 → 邻接顶点 ----
        auto vtx2vtx_pair = vtx2vtx_from_polygon_edges_with_vtx2elem(
            elem2idx, idx2vtx, vtx2jdx, jdx2elem, false);

        const vector<size_t> &vtx2idx = vtx2vtx_pair.first;
        const vector<size_t> &idx2adj = vtx2vtx_pair.second;

        // ---- 3. 转换成 edge2vtx（每条边两个端点）----
        return edge2vtx_from_vtx2vtx(vtx2idx, idx2adj);
    }

    pair<vector<size_t>, vector<size_t>> vtx2elem_from_polygon_mesh(
        const vector<size_t> &elem2idx,
        const vector<size_t> &idx2vtx,
        size_t num_vtx)
    {
        size_t num_elem = elem2idx.size() - 1;

        // 每个顶点参与了多少元素（prefix sum）
        vector<size_t> vtx2jdx(num_vtx + 1, 0);

        for (size_t e = 0; e < num_elem; e++)
        {
            for (size_t k = elem2idx[e]; k < elem2idx[e + 1]; k++)
            {
                size_t v = idx2vtx[k];
                vtx2jdx[v + 1] += 1; // 统计 v 的个数
            }
        }

        // 前缀和构建 index
        for (size_t v = 0; v < num_vtx; v++)
        {
            vtx2jdx[v + 1] += vtx2jdx[v];
        }

        size_t total = vtx2jdx[num_vtx];
        vector<size_t> jdx2elem(total);

        // 填充 jdx2elem
        vector<size_t> vtx2jdx_copy = vtx2jdx;
        for (size_t e = 0; e < num_elem; e++)
        {
            for (size_t k = elem2idx[e]; k < elem2idx[e + 1]; k++)
            {
                size_t v = idx2vtx[k];
                size_t j = vtx2jdx_copy[v];
                jdx2elem[j] = e;
                vtx2jdx_copy[v] += 1;
            }
        }

        return {vtx2jdx, jdx2elem};
    }

    pair<vector<size_t>, vector<size_t>> vtx2vtx_from_polygon_edges_with_vtx2elem(
        const vector<size_t> &elem2idx,
        const vector<size_t> &idx2vtx,
        const vector<size_t> &vtx2jdx,
        const vector<size_t> &jdx2elem,
        bool is_bidirectional)
    {
        size_t nvtx = vtx2jdx.size() - 1;

        vector<size_t> vtx2kdx(nvtx + 1, 0);
        vector<size_t> kdx2vtx;

        for (size_t v = 0; v < nvtx; v++)
        {

            set<size_t> adj; // 用 set 避免重复

            // 遍历所有包含 v 的 polygon
            for (size_t di = vtx2jdx[v]; di < vtx2jdx[v + 1]; di++)
            {
                size_t elem = jdx2elem[di];

                size_t begin = elem2idx[elem];
                size_t end = elem2idx[elem + 1];
                size_t n = end - begin;

                // polygon 中所有边
                for (size_t e = 0; e < n; e++)
                {
                    size_t v0 = idx2vtx[begin + e];
                    size_t v1 = idx2vtx[begin + ((e + 1) % n)];

                    if (v0 != v && v1 != v)
                        continue;

                    size_t other = (v0 == v ? v1 : v0);

                    if (is_bidirectional || other > v)
                        adj.insert(other);
                }
            }

            for (auto to : adj)
                kdx2vtx.push_back(to);

            vtx2kdx[v + 1] = kdx2vtx.size();
        }

        return {vtx2kdx, kdx2vtx};
    }

    std::vector<size_t> elem2elem_from_polygon_mesh_with_vtx2elem(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const std::vector<size_t> &vtx2jdx,
        const std::vector<size_t> &jdx2elem)
    {
        assert(!vtx2jdx.empty());

        size_t num_elem = elem2idx.size() - 1;
        std::vector<size_t> idx2elem(idx2vtx.size(), INVALID);

        for (size_t i_elem = 0; i_elem < num_elem; ++i_elem)
        {
            size_t start_i = elem2idx[i_elem];
            size_t end_i = elem2idx[i_elem + 1];
            size_t num_edge_i = end_i - start_i;

            for (size_t i_edge = 0; i_edge < num_edge_i; ++i_edge)
            {
                size_t i0 = idx2vtx[start_i + i_edge];
                size_t i1 = idx2vtx[start_i + (i_edge + 1) % num_edge_i];

                // 遍历所有包含 i0 的 polygon
                for (size_t p = vtx2jdx[i0]; p < vtx2jdx[i0 + 1]; ++p)
                {
                    size_t j_elem = jdx2elem[p];
                    if (j_elem == i_elem)
                        continue;

                    size_t start_j = elem2idx[j_elem];
                    size_t end_j = elem2idx[j_elem + 1];
                    size_t num_edge_j = end_j - start_j;

                    for (size_t j_edge = 0; j_edge < num_edge_j; ++j_edge)
                    {
                        size_t j0 = idx2vtx[start_j + j_edge];
                        size_t j1 = idx2vtx[start_j + (j_edge + 1) % num_edge_j];

                        // 反向边匹配
                        if (i0 == j1 && i1 == j0)
                        {
                            idx2elem[start_i + i_edge] = j_elem;
                            goto FOUND;
                        }
                    }
                }
            FOUND:
                continue;
            }
        }

        return idx2elem;
    }

    std::vector<size_t> elem2elem_from_polygon_mesh(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        size_t num_vtx)
    {
        // 1) build vtx -> elem adjacency
        auto vtx2elem = vtx2elem_from_polygon_mesh(
            elem2idx,
            idx2vtx,
            num_vtx);

        const std::vector<size_t> &vtx2jdx = vtx2elem.first;
        const std::vector<size_t> &jdx2elem = vtx2elem.second;

        // 2) build elem -> elem adjacency
        return elem2elem_from_polygon_mesh_with_vtx2elem(
            elem2idx,
            idx2vtx,
            vtx2jdx,
            jdx2elem);
    }

    // elem_from_polygon_mesh_as_points
    // - elem2idx: size = num_elem + 1
    // - idx2vtx: flattened vertex indices
    // - vtx2xyz: flattened vertex coordinates (num_vtx * num_dim)
    // - num_dim: dimension per vertex (2 or 3)
    //
    // return: flattened elem2cog (num_elem * num_dim)

    std::vector<float> polyMesh_elem2area(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const float *vtx2xy)
    {
        const size_t num_elem = elem2idx.size() - 1;
        std::vector<float> areas(num_elem, 0.0f);

        for (size_t i_elem = 0; i_elem < num_elem; ++i_elem)
        {
            const size_t num_vtx_in_elem =
                elem2idx[i_elem + 1] - elem2idx[i_elem];

            for (size_t i_edge = 0; i_edge < num_vtx_in_elem; ++i_edge)
            {
                const size_t i0_vtx =idx2vtx[elem2idx[i_elem] + i_edge];

                const size_t i1_vtx =idx2vtx[elem2idx[i_elem] +(i_edge + 1) % num_vtx_in_elem];

                areas[i_elem] +=0.5f * vtx2xy[i0_vtx * 2 + 0] *vtx2xy[i1_vtx * 2 + 1];

                areas[i_elem] -=0.5f * vtx2xy[i0_vtx * 2 + 1] *vtx2xy[i1_vtx * 2 + 0];
            }
        }

        return areas;
    }
}