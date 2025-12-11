#pragma once

#include <vector>
#include <algorithm>
#include <set>



namespace polygonMesh
{

    using std::vector, std::pair, std::set;
    
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
        const vector<Index> &idx2vtx);

};