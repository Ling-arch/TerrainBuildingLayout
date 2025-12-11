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

    template <typename Index>
    vector<Index> edge2vtx_from_vtx2vtx(
        const vector<Index> &vtx2idx,
        const vector<Index> &idx2vtx)
    {
        vector<Index> line2vtx;
        line2vtx.reserve(idx2vtx.size() * 2);

        for (size_t v = 0; v + 1 < vtx2idx.size(); v++)
        {
            size_t begin = vtx2idx[v];
            size_t end = vtx2idx[v + 1];

            for (size_t k = begin; k < end; k++)
            {
                Index u = idx2vtx[k];
                line2vtx.push_back(static_cast<Index>(v));
                line2vtx.push_back(u);
            }
        }
        return line2vtx;
    }

}