#pragma once

#include <vector>
#include <algorithm>
#include <set>
#include <limits>
#include <cassert>

namespace polygonMesh
{

    using std::vector, std::pair, std::set;

    static constexpr size_t INVALID = static_cast<size_t>(std::numeric_limits<int64_t>::max());

    std::vector<size_t> elem2elem_from_polygon_mesh(const std::vector<size_t> &elem2idx,const std::vector<size_t> &idx2vtx,size_t num_vtx);

    std::vector<size_t> elem2elem_from_polygon_mesh_with_vtx2elem(
            const std::vector<size_t> &elem2idx,
            const std::vector<size_t> &idx2vtx,
            const std::vector<size_t> &vtx2jdx,
            const std::vector<size_t> &jdx2elem);

    vector<size_t> edge2vtx_from_polygon_mesh(
        const vector<size_t> &elem2idx,
        const vector<size_t> &idx2vtx,
        size_t num_vtx);

    // 输入：
    //  elem2idx[i]..elem2idx[i+1] 为第 i 个 polygon 的顶点范围
    //  idx2vtx 是所有多边形顶点的线性数组
    //  num_vtx 顶点数量
    //
    // 输出：
    //  (vtx2jdx, jdx2elem)
    //  顶点 v 的所有相邻 polygon 在 jdx2elem 中的区间为：
    //     jdx2elem[vtx2jdx[v] .. vtx2jdx[v+1]-1]
    pair<vector<size_t>, vector<size_t>> vtx2elem_from_polygon_mesh(
        const vector<size_t> &elem2idx,
        const vector<size_t> &idx2vtx,
        size_t num_vtx);

    pair<vector<size_t>, vector<size_t>> vtx2vtx_from_polygon_edges_with_vtx2elem(
        const vector<size_t> &elem2idx,
        const vector<size_t> &idx2vtx,
        const vector<size_t> &vtx2jdx,
        const vector<size_t> &jdx2elem,
        bool is_bidirectional);

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

    template <typename ElemFace2AdjElem>
    void elem2group_mark_connected_elements_for_polygon_mesh(
        std::vector<size_t> &elem2group,
        size_t idx_elem_kernel,
        size_t idx_group,
        const std::vector<size_t> &elem2idx,
        ElemFace2AdjElem elemface2adjelem)
    {
        const size_t num_elem = elem2group.size();
        assert(num_elem + 1 == elem2idx.size());

        // mark kernel
        elem2group[idx_elem_kernel] = idx_group;

        // stack (DFS) — Rust 里是 Vec + pop
        std::vector<size_t> next;
        next.push_back(idx_elem_kernel);

        while (!next.empty())
        {
            size_t i_elem0 = next.back();
            next.pop_back();

            size_t num_adj = elem2idx[i_elem0 + 1] - elem2idx[i_elem0];

            for (size_t i_face0 = 0; i_face0 < num_adj; ++i_face0)
            {
                size_t j_elem = elemface2adjelem(i_elem0, i_face0);
                if (j_elem == INVALID)
                    continue;

                if (elem2group[j_elem] != idx_group)
                {
                    elem2group[j_elem] = idx_group;
                    next.push_back(j_elem);
                }
            }
        }
    }

    template <typename ElemFace2AdjElem>
    std::pair<size_t, std::vector<size_t>>
    elem2group_from_polygon_mesh(
        const std::vector<size_t> &elem2idx,
        ElemFace2AdjElem elemface2adjelem)
    {
        const size_t nelem = elem2idx.size() - 1;

        std::vector<size_t> elem2group(nelem, INVALID);

        size_t i_group = 0;

        while (true)
        {
            size_t kernel = INVALID;

            // find first ungrouped element
            for (size_t i = 0; i < nelem; ++i)
            {
                if (elem2group[i] == INVALID)
                {
                    kernel = i;
                    break;
                }
            }

            if (kernel == INVALID)
                break;

            elem2group_mark_connected_elements_for_polygon_mesh(
                elem2group,
                kernel,
                i_group,
                elem2idx,
                elemface2adjelem);

            ++i_group;
        }

        return {i_group, elem2group};
    }

    template <typename T>
    std::vector<T> elem2center_from_polygon_mesh_as_points(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const std::vector<T> &vtx2xyz,
        size_t num_dim)
    {
        static_assert(std::is_floating_point<T>::value,
                      "T must be float or double");

        const size_t num_elem = elem2idx.size() - 1;

        std::vector<T> cog(num_dim, T(0));
        std::vector<T> elem2cog;
        elem2cog.reserve(num_elem * num_dim);

        for (size_t i_elem = 0; i_elem < num_elem; ++i_elem)
        {
            // reset cog
            std::fill(cog.begin(), cog.end(), T(0));

            const size_t start = elem2idx[i_elem];
            const size_t end = elem2idx[i_elem + 1];
            const size_t num_vtx_in_elem = end - start;

            // accumulate vertex positions
            for (size_t k = start; k < end; ++k)
            {
                size_t i_vtx = idx2vtx[k];
                for (size_t d = 0; d < num_dim; ++d)
                {
                    cog[d] += vtx2xyz[i_vtx * num_dim + d];
                }
            }

            const T ratio = (num_vtx_in_elem == 0)
                                ? T(0)
                                : T(1) / T(num_vtx_in_elem);

            // write result
            for (size_t d = 0; d < num_dim; ++d)
            {
                elem2cog.push_back(cog[d] * ratio);
            }
        }

        return elem2cog;
    }

    std::vector<float> polyMesh_elem2area(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const float *vtx2xy);
}