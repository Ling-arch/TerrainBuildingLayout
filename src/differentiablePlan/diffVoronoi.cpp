#include "diffVoronoi.h"
#include <stdexcept>
#include <limits>
#include <iostream>

namespace diffVoronoi
{
    static constexpr size_t SIZE_MAX_T = static_cast<size_t>(std::numeric_limits<int64_t>::max());

    Vector2 get_tensor_row_to_vec2(const float *ptr, size_t i)
    {
        return Vector2(ptr[2 * i + 0], ptr[2 * i + 1]);
    }

    std::vector<float> flat_tensor_to_float(const torch::Tensor &t)
    {
        TORCH_CHECK(t.device().is_cpu(), "tensor must be on CPU");
        TORCH_CHECK(t.dim() == 2 && t.size(1) == 2, "expected Nx2 tensor");
        auto cont = t.contiguous();
        auto ptr = cont.data_ptr<float>();
        std::vector<float> out;
        out.reserve(cont.size(0) * 2);
        for (size_t i = 0; i < cont.size(0) * 2; ++i)
            out.push_back(ptr[i]);
        return out;
    }

    torch::Tensor vec2_to_tensor(const std::vector<Vector2> &list)
    {
        // 提前分配内存存储 x/y 数据，避免多次拷贝
        std::vector<float> data;
        data.reserve(list.size() * 2);
        for (const auto &vec : list)
        {
            data.push_back(vec.x());
            data.push_back(vec.y());
        }
        // 显式配置张量选项
        torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        // 从内存直接创建张量（无额外拷贝），形状用显式转换保证类型正确
        // clone 确保张量拥有数据所有权（否则 data 析构后张量指针失效）
        torch::Tensor t = torch::from_blob(data.data(), {static_cast<int64_t>(list.size()), 2}, opts).clone();
        return t;
    }

    torch::Tensor flatvec_to_tensor(const std::vector<float> &data)
    {
        if (data.empty())
        {
            // 返回空张量
            return torch::empty({0, 2}, torch::kFloat32);
        }

        // 确保数据大小是偶数
        if (data.size() % 2 != 0)
        {
            throw std::runtime_error("flatvec_to_tensor: data size must be even (x,y pairs)");
        }

        // 方法1：使用 clone() 确保所有权
        torch::TensorOptions opts = torch::TensorOptions()
                                        .dtype(torch::kFloat32)
                                        .device(torch::kCPU);

        // 注意：先转换为 int64_t，再除以2
        int64_t rows = static_cast<int64_t>(data.size()) / 2;

        torch::Tensor t = torch::from_blob(
                              const_cast<float *>(data.data()), // from_blob 需要非 const 指针
                              {rows, 2},
                              opts)
                              .clone(); // clone 创建独立副本

        return t;
    }

    // ---- VoronoiFunction::forward
    torch::Tensor VoronoiFunction::forward(torch::autograd::AutogradContext *ctx,
                                           const torch::Tensor &site2xy,
                                           const std::vector<float> &vtxl2xy,
                                           const std::vector<std::array<size_t, 4>> &vtxv2info)
    {
        TORCH_CHECK(site2xy.device().is_cpu(), "site2xy must be on CPU");
        TORCH_CHECK(site2xy.dtype() == torch::kFloat32, "site2xy must be float32");
        TORCH_CHECK(site2xy.dim() == 2 && site2xy.size(1) == 2, "site2xy must be (N,2)");

        torch::Tensor vtxl2xy_tensor = flatvec_to_tensor(vtxl2xy);
        // Save site2xy and vtxl2xy tensor for backward
        ctx->save_for_backward({site2xy, vtxl2xy_tensor});
        std::vector<std::vector<int64_t>> vtxv2info_vec;
        vtxv2info_vec.reserve(vtxv2info.size());
        for (const auto &a : vtxv2info)
        {
            vtxv2info_vec.push_back({static_cast<int64_t>(a[0]),
                                     static_cast<int64_t>(a[1]),
                                     static_cast<int64_t>(a[2]),
                                     static_cast<int64_t>(a[3])});
        }
        ctx->saved_data["vtxv2info"] = vtxv2info_vec;

        // Prepare site2xy as vector<double> flattened for calling voronoi2 functions
        const size_t num_site = site2xy.size(0);
        const float *site_ptr = site2xy.data_ptr<float>();
        std::vector<float> site_flat;
        site_flat.resize(num_site * 2);
        for (size_t i = 0; i < num_site; ++i)
        {
            site_flat[2 * i + 0] = site_ptr[2 * i + 0];
            site_flat[2 * i + 1] = site_ptr[2 * i + 1];
        }
        // compute vtxv positions
        size_t num_vtxv = vtxv2info.size();
        std::vector<Vector2> out;
        out.resize(num_vtxv, Vector2::Zero());
        for (size_t i = 0; i < num_vtxv; ++i)
        {
            out[i] = voronoi2::position_of_voronoi_vertex(vtxv2info[i], vtxl2xy, site_flat);
        }

        return vec2_to_tensor(out);
    }

    // ---- VoronoiFunction::backward
    torch::autograd::tensor_list VoronoiFunction::backward(torch::autograd::AutogradContext *ctx,
                                                           torch::autograd::tensor_list grad_outputs)
    {
        // grad_outputs[0] is gradient wrt forward output vtxv2xy (M,2) float32
        TORCH_CHECK(grad_outputs.size() >= 1, "expected grad_outputs[0]");

        auto saved = ctx->get_saved_variables();
        TORCH_CHECK(saved.size() >= 1, "saved variables missing");
        torch::Tensor site2xy = saved[0];
        TORCH_CHECK(site2xy.device().is_cpu(), "site2xy must be CPU");
        TORCH_CHECK(site2xy.dtype() == torch::kFloat32, "site2xy must be float32");

        // recover params

        auto iv_vtxv2info = ctx->saved_data["vtxv2info"];

        torch::Tensor vtxl2xy_tensor = saved[1];
        std::vector<float> vtxl2xy(
            vtxl2xy_tensor.data_ptr<float>(),
            vtxl2xy_tensor.data_ptr<float>() + vtxl2xy_tensor.numel());

        // convert vtxv2info back
        std::vector<std::array<size_t, 4>> vtxv2info;
        {
            auto outer = iv_vtxv2info.toList();
            vtxv2info.reserve(outer.size());
            for (size_t i = 0; i < outer.size(); ++i)
            {
                auto inner = outer.get(i).toIntVector(); // returns vector<size_t>
                std::array<size_t, 4> arr = {static_cast<size_t>(inner[0]),
                                             static_cast<size_t>(inner[1]),
                                             static_cast<size_t>(inner[2]),
                                             static_cast<size_t>(inner[3])};
                vtxv2info.push_back(arr);
            }
        }

        // load site2xy into vector<Vector2d>
        const size_t num_site = site2xy.size(0);
        const float *site_ptr = site2xy.data_ptr<float>();
        std::vector<Vector2> site_vec;
        site_vec.resize(num_site);
        for (size_t i = 0; i < num_site; ++i)
            site_vec[i] = get_tensor_row_to_vec2(site_ptr, i);

        // load grad_outputs[0] into vector<Vector2d> dv
        torch::Tensor dv_tensor = grad_outputs[0];
        TORCH_CHECK(dv_tensor.dim() == 2 && dv_tensor.size(1) == 2, "grad output must be (M,2)");
        TORCH_CHECK(dv_tensor.dtype() == torch::kFloat32, "grad output must be float32");
        const size_t num_vtxv = dv_tensor.size(0);
        const float *dv_ptr = dv_tensor.data_ptr<float>();
        std::vector<Vector2> dv_vec;
        dv_vec.resize(static_cast<size_t>(num_vtxv));
        for (size_t i = 0; i < num_vtxv; ++i)
            dv_vec[i] = get_tensor_row_to_vec2(dv_ptr, i);

        // accumulate dw_site2xy
        std::vector<float> dw_site2xy_flat(static_cast<size_t>(num_site) * 2, 0.f);

        for (size_t i_vtxv = 0; i_vtxv < num_vtxv; ++i_vtxv)
        {
            const auto &info = vtxv2info[i_vtxv];
            if (info[1] == SIZE_MAX_T)
            {
                // original polygon vertex -> no grad w.r.t. sites
                // std::cout << "original polygon vertex index is " << i_vtxv << std::endl;
                continue;
            }
            else if (info[3] == SIZE_MAX_T)
            {
                // std::cout << "polygon edge with two sites vertex index is " << i_vtxv << std::endl;
                //  intersection of loop edge and two sites
                size_t num_vtxl = vtxl2xy.size() / 2;
                size_t i1_loop = info[0];
                if (!(i1_loop < num_vtxl))
                    throw std::runtime_error("loop index out of range");
                size_t i2_loop = (i1_loop + 1) % num_vtxl;

                Vector2 l1(vtxl2xy[2 * i1_loop], vtxl2xy[2 * i1_loop + 1]);
                Vector2 l2(vtxl2xy[2 * i2_loop], vtxl2xy[2 * i2_loop + 1]);

                size_t i0_site = info[1];
                size_t i1_site = info[2];
                Vector2 s0 = site_vec[i0_site];
                Vector2 s1 = site_vec[i1_site];
                // std::cout << "Two sites is (" << i0_site << "," << i1_site << ")" << std::endl;
                //  call utility: returns r, dr/ds0 (Matrix2d), dr/ds1 (Matrix2d)
                Vector2 r;
                Matrix2 drds0, drds1;
                std::tie(r, drds0, drds1) = M2::dw_intersection_against_bisector(l1, (l2 - l1), s0, s1);

                Vector2 dv = dv_vec[i_vtxv];

                Vector2 ds0 = drds0.transpose() * dv;
                Vector2 ds1 = drds1.transpose() * dv;

                dw_site2xy_flat[i0_site * 2 + 0] += ds0.x();
                dw_site2xy_flat[i0_site * 2 + 1] += ds0.y();
                dw_site2xy_flat[i1_site * 2 + 0] += ds1.x();
                dw_site2xy_flat[i1_site * 2 + 1] += ds1.y();
            }
            else
            {
                // std::cout << "three sites vtxv index is " << i_vtxv << std::endl;
                //  circumcenter of three sites
                size_t idx0 = info[1];
                size_t idx1 = info[2];
                size_t idx2 = info[3];
                // std::cout << "Three sites is (" << idx0 << ", " << idx1 << ", " << idx2 << ")" << std::endl;
                Vector2 s0 = site_vec[idx0];
                Vector2 s1 = site_vec[idx1];
                Vector2 s2 = site_vec[idx2];

                auto circumResult = M2::wdw_circumcenter(s0, s1, s2);
                std::array<Matrix2, 3> dvds = circumResult.dcc;

                Vector2 dv = dv_vec[i_vtxv];
                for (size_t i_node = 0; i_node < 3; ++i_node)
                {
                    Vector2 ds = dvds[i_node].transpose() * dv;
                    size_t is0 = (i_node == 0 ? idx0 : (i_node == 1 ? idx1 : idx2));
                    dw_site2xy_flat[is0 * 2 + 0] += ds.x();
                    dw_site2xy_flat[is0 * 2 + 1] += ds.y();
                }
            }
        }
        // convert dw_site2xy_flat -> torch tensor float32 same shape as site2xy

        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor grad_site = torch::zeros({static_cast<int64_t>(num_site), 2}, opts);
        float *gptr = grad_site.data_ptr<float>();
        for (size_t i = 0; i < num_site; ++i)
        {
            gptr[2 * i + 0] = static_cast<float>(dw_site2xy_flat[2 * i + 0]);
            gptr[2 * i + 1] = static_cast<float>(dw_site2xy_flat[2 * i + 1]);
        }

        // return gradient for first arg (site2xy), and nullptrs for other (vtxl2xy/vtxv2info) which are constants
        return {grad_site, torch::Tensor(), torch::Tensor()};
    }

    std::tuple<torch::Tensor, VoronoiInfo, std::vector<voronoi2::Cell>> voronoi(
        const std::vector<float> &vtxl2xy_f,
        const torch::Tensor &site2xy,
        const std::function<bool(size_t)> &site2isalive)
    {
        // 1) flatten site tensor
        std::vector<float> site_flat = flat_tensor_to_float(site2xy);

        // 2) compute site2cell
        std::vector<voronoi2::Cell> site2cell = voronoi2::voronoi_cells(vtxl2xy_f, site_flat, site2isalive);

        // 3) indexing
        voronoi2::VoronoiMesh voronoi_mesh = voronoi2::indexing(site2cell);

        // 4) autograd layer
        VoronoiLayer layer(std::vector<float>(vtxl2xy_f.begin(), vtxl2xy_f.end()), voronoi_mesh.vtxv2info);

        torch::Tensor vtxv2xy = layer.forward(site2xy);

        //   -----idx2site（elem2elem）
        size_t num_vtxv = vtxv2xy.size(0);

        std::vector<size_t> idx2site = polygonMesh::elem2elem_from_polygon_mesh(
            voronoi_mesh.site2idx,
            voronoi_mesh.idx2vtxv,
            num_vtxv);

        // 5) pack VoronoiInfo
        VoronoiInfo vi;
        vi.site2idx = voronoi_mesh.site2idx;
        vi.idx2vtxv = voronoi_mesh.idx2vtxv;
        vi.vtxv2info = voronoi_mesh.vtxv2info;
        vi.idx2site = std::move(idx2site);

        return std::make_tuple(vtxv2xy, vi, site2cell);
    }

    torch::Tensor PolygonMesh2ToCogsFunction::forward(torch::autograd::AutogradContext *ctx,
                                                      const torch::Tensor &vtx2xy,
                                                      const std::vector<size_t> &elem2idx,
                                                      const std::vector<size_t> &idx2vtx)
    {
        TORCH_CHECK(vtx2xy.device().is_cpu(), "vtx2xy must be CPU");
        TORCH_CHECK(vtx2xy.dim() == 2 && vtx2xy.size(1) == 2,
                    "vtx2xy must be (N,2)");

        torch::Tensor vcont = vtx2xy.contiguous();
        const float *vptr = vcont.data_ptr<float>();

        std::vector<float> elem2cog =
            polygonMesh::elem2center_from_polygon_mesh_as_points<float>(
                elem2idx,
                idx2vtx,
                std::vector<float>(vptr, vptr + vtx2xy.numel()),
                2);

        const int64_t num_elem = static_cast<int64_t>(elem2idx.size() - 1);

        torch::Tensor out = torch::from_blob(
                                elem2cog.data(),
                                {num_elem, 2},
                                torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                                .clone(); // clone 保证内存安全

        // ===== 保存 backward 所需 =====
        ctx->save_for_backward({vtx2xy});
        std::vector<int64_t> elem2idx_arr;
        std::vector<int64_t> idx2vtx_arr;
        elem2idx_arr.reserve(elem2idx.size());
        idx2vtx_arr.reserve(idx2vtx.size());
        for (auto &idx : elem2idx)
        {
            elem2idx_arr.push_back(static_cast<int64_t>(idx));
        }
        for (auto &idx : idx2vtx)
        {
            idx2vtx_arr.push_back(static_cast<int64_t>(idx));
        }
        ctx->saved_data["elem2idx"] = elem2idx_arr;
        ctx->saved_data["idx2vtx"] = idx2vtx_arr;

        return out;
    }

    torch::autograd::tensor_list PolygonMesh2ToCogsFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        TORCH_CHECK(grad_outputs.size() == 1);

        auto dw_elem2cog = grad_outputs[0];
        TORCH_CHECK(dw_elem2cog.device().is_cpu());
        TORCH_CHECK(dw_elem2cog.dim() == 2 && dw_elem2cog.size(1) == 2);

        auto saved = ctx->get_saved_variables();
        auto vtx2xy = saved[0];

        const auto &elem2idx = ctx->saved_data["elem2idx"].toIntVector();
        const auto &idx2vtx = ctx->saved_data["idx2vtx"].toIntVector();

        const int64_t num_vtx = vtx2xy.size(0);

        auto grad_vtx = torch::zeros(
            {num_vtx, 2},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        auto gptr = dw_elem2cog.contiguous().data_ptr<float>();
        auto acc = grad_vtx.accessor<float, 2>();

        const size_t num_elem = elem2idx.size() - 1;

        // ===== 完全对应 Rust bwd =====
        for (size_t i_elem = 0; i_elem < num_elem; ++i_elem)
        {
            size_t start = static_cast<size_t>(elem2idx[i_elem]);
            size_t end = static_cast<size_t>(elem2idx[i_elem + 1]);
            size_t nv = end - start;

            if (nv == 0)
                continue;

            float ratio = 1.0f / float(nv);
            float gx = gptr[i_elem * 2 + 0];
            float gy = gptr[i_elem * 2 + 1];

            for (size_t k = start; k < end; ++k)
            {
                size_t i_vtx = static_cast<size_t>(idx2vtx[k]);
                acc[i_vtx][0] += ratio * gx;
                acc[i_vtx][1] += ratio * gy;
            }
        }

        // 只有 vtx2xy 有梯度，其它是常量
        return {grad_vtx, torch::Tensor(), torch::Tensor()};
    }

    torch::Tensor PolygonMesh2ToAreaFunction::forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &vtx2xy,
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx)
    {
        TORCH_CHECK(vtx2xy.device().is_cpu(), "vtx2xy must be CPU");
        TORCH_CHECK(vtx2xy.dim() == 2 && vtx2xy.size(1) == 2);

        auto vcont = vtx2xy.contiguous();
        const float *vptr = vcont.data_ptr<float>();

        // === 调用elem2area
        std::vector<float> areas = polygonMesh::polyMesh_elem2area(elem2idx, idx2vtx, vptr);

        auto out = torch::from_blob(
                       areas.data(),
                       {(int64_t)areas.size()},
                       torch::TensorOptions()
                           .dtype(torch::kFloat32)
                           .device(torch::kCPU))
                       .clone();

        // backward 需要的数据
        ctx->save_for_backward({vtx2xy});
        std::vector<int64_t> elem2idx_arr;
        std::vector<int64_t> idx2vtx_arr;
        elem2idx_arr.reserve(elem2idx.size());
        idx2vtx_arr.reserve(idx2vtx.size());
        for (auto &idx : elem2idx)
        {
            elem2idx_arr.push_back(static_cast<int64_t>(idx));
        }
        for (auto &idx : idx2vtx)
        {
            idx2vtx_arr.push_back(static_cast<int64_t>(idx));
        }
        ctx->saved_data["elem2idx"] = elem2idx_arr;
        ctx->saved_data["idx2vtx"] = idx2vtx_arr;

        return out;
    }

    torch::autograd::tensor_list PolygonMesh2ToAreaFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        TORCH_CHECK(grad_outputs.size() == 1);

        auto dw_area = grad_outputs[0];
        TORCH_CHECK(dw_area.device().is_cpu());
        TORCH_CHECK(dw_area.dim() == 1);

        auto saved = ctx->get_saved_variables();
        auto vtx2xy = saved[0];

        const auto &elem2idx = ctx->saved_data["elem2idx"].toIntVector();
        const auto &idx2vtx = ctx->saved_data["idx2vtx"].toIntVector();

        const int64_t num_vtx = vtx2xy.size(0);

        auto grad_vtx = torch::zeros(
            {num_vtx, 2},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        const float *vptr = vtx2xy.contiguous().data_ptr<float>();
        const float *gptr = dw_area.contiguous().data_ptr<float>();
        auto acc = grad_vtx.accessor<float, 2>();

        const size_t num_elem = elem2idx.size() - 1;

        for (size_t i_elem = 0; i_elem < num_elem; ++i_elem)
        {
            float w = gptr[i_elem];
            size_t start = static_cast<size_t>(elem2idx[i_elem]);
            size_t end = static_cast<size_t>(elem2idx[i_elem + 1]);
            size_t nv = end - start;

            for (size_t i = 0; i < nv; ++i)
            {
                size_t i0 = static_cast<size_t>(idx2vtx[start + i]);
                size_t i1 = static_cast<size_t>(idx2vtx[start + (i + 1) % nv]);

                acc[i0][0] += 0.5f * vptr[i1 * 2 + 1] * w;
                acc[i1][1] += 0.5f * vptr[i0 * 2 + 0] * w;
                acc[i0][1] -= 0.5f * vptr[i1 * 2 + 0] * w;
                acc[i1][0] -= 0.5f * vptr[i0 * 2 + 1] * w;
            }
        }

        return {
            grad_vtx,
            torch::Tensor(),
            torch::Tensor()};
    }

    torch::Tensor Vtx2XYZToEdgeVectorFunction::forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &vtx2xy,        // (num_vtx, num_dim)
        const std::vector<size_t> &edge2vtx // len = num_edge * 2
    )
    {
        TORCH_CHECK(vtx2xy.device().is_cpu());
        TORCH_CHECK(vtx2xy.dtype() == torch::kFloat32);
        TORCH_CHECK(vtx2xy.dim() == 2);

        const auto num_vtx = vtx2xy.size(0);
        const auto num_dim = vtx2xy.size(1);
        const size_t num_edge = edge2vtx.size() / 2;

        auto vtx = vtx2xy.contiguous();
        const float *vtx_ptr = vtx.data_ptr<float>();

        torch::Tensor edge2xy = torch::zeros({static_cast<int64_t>(num_edge), num_dim}, vtx.options());

        float *edge_ptr = edge2xy.data_ptr<float>();

        for (size_t i_edge = 0; i_edge < num_edge; ++i_edge)
        {
            size_t i0_vtx = edge2vtx[i_edge * 2 + 0];
            size_t i1_vtx = edge2vtx[i_edge * 2 + 1];

            for (int d = 0; d < num_dim; ++d)
            {
                edge_ptr[i_edge * num_dim + d] += vtx_ptr[i1_vtx * num_dim + d];
                edge_ptr[i_edge * num_dim + d] -= vtx_ptr[i0_vtx * num_dim + d];
            }
        }

        // backward 需要 edge2vtx 和 num_vtx
        ctx->save_for_backward({});
        std::vector<int64_t> edge2vtx_arr;
        edge2vtx_arr.reserve(edge2vtx.size());
        for (auto &vtx_idx : edge2vtx)
            edge2vtx_arr.push_back(static_cast<int64_t>(vtx_idx));
        ctx->saved_data["edge2vtx"] = edge2vtx_arr;
        ctx->saved_data["num_vtx"] = (int64_t)num_vtx;
        ctx->saved_data["num_dim"] = (int64_t)num_dim;

        return edge2xy;
    }

    torch::autograd::tensor_list Vtx2XYZToEdgeVectorFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        auto dw_edge2xy = grad_outputs[0];
        TORCH_CHECK(dw_edge2xy.defined());

        const auto edge2vtx = ctx->saved_data["edge2vtx"].toIntVector();
        const int64_t num_vtx = ctx->saved_data["num_vtx"].toInt();
        const int64_t num_dim = ctx->saved_data["num_dim"].toInt();

        const size_t num_edge = edge2vtx.size() / 2;

        auto dw_edge = dw_edge2xy.contiguous();
        const float *dw_edge_ptr = dw_edge.data_ptr<float>();

        torch::Tensor dw_vtx2xy = torch::zeros({num_vtx, num_dim}, dw_edge.options());

        float *dw_vtx_ptr = dw_vtx2xy.data_ptr<float>();

        for (size_t i_edge = 0; i_edge < num_edge; ++i_edge)
        {
            size_t i0 = edge2vtx[i_edge * 2 + 0];
            size_t i1 = edge2vtx[i_edge * 2 + 1];

            for (int d = 0; d < num_dim; ++d)
            {
                float g = dw_edge_ptr[i_edge * num_dim + d];
                dw_vtx_ptr[i1 * num_dim + d] += g;
                dw_vtx_ptr[i0 * num_dim + d] -= g;
            }
        }

        // 返回顺序必须和 forward 参数一致
        return {
            dw_vtx2xy,      // grad vtx2xy
            torch::Tensor() // grad edge2vtx (None)
        };
    }

    torch::Tensor EdgeTensorAlignFunction::forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &vtx2xy,
        const std::vector<size_t> &edge2vtx,
        const torch::Tensor &theta_dir)
    {
        TORCH_CHECK(vtx2xy.device().is_cpu());
        TORCH_CHECK(vtx2xy.dtype() == torch::kFloat32);
        TORCH_CHECK(theta_dir.dtype() == torch::kFloat32);

        const int64_t num_edge = edge2vtx.size() / 2;

        auto vtx = vtx2xy.contiguous();
        auto tdir = theta_dir.contiguous();

        const float *vtx_ptr = vtx.data_ptr<float>();
        const float *tdir_ptr = tdir.data_ptr<float>();

        torch::Tensor loss = torch::zeros({}, vtx.options());
        float total = 0.f;

        for (int64_t e = 0; e < num_edge; ++e)
        {
            size_t i0 = edge2vtx[2 * e + 0];
            size_t i1 = edge2vtx[2 * e + 1];

            float ex = vtx_ptr[i1 * 2 + 0] - vtx_ptr[i0 * 2 + 0];
            float ey = vtx_ptr[i1 * 2 + 1] - vtx_ptr[i0 * 2 + 1];

            float phi = std::atan2(ey, ex);

            float tx = tdir_ptr[e * 2 + 0];
            float ty = tdir_ptr[e * 2 + 1];
            float theta = std::atan2(ty, tx);

            float d = phi - theta;
            if (d > M_PI)
                d -= 2.f * M_PI;
            if (d < -M_PI)
                d += 2.f * M_PI;

            total += 1.f - std::cos(4.f * d);
        }

        loss.fill_(total / float(num_edge));

        // ===== 保存给 backward =====
        ctx->saved_data["edge2vtx"] =
            std::vector<int64_t>(edge2vtx.begin(), edge2vtx.end());
        ctx->save_for_backward({vtx, tdir});

        return loss;
    }

    torch::autograd::tensor_list
    EdgeTensorAlignFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        const auto grad_out = grad_outputs[0]; // scalar

        const auto edge2vtx = ctx->saved_data["edge2vtx"].toIntVector();
        const int64_t num_edge = edge2vtx.size() / 2;

        auto saved = ctx->get_saved_variables();
        auto vtx = saved[0];
        auto tdir = saved[1];

        const float *vtx_ptr = vtx.data_ptr<float>();
        const float *tdir_ptr = tdir.data_ptr<float>();

        torch::Tensor grad_vtx = torch::zeros_like(vtx);
        float *gv = grad_vtx.data_ptr<float>();

        const float gscale = grad_out.item<float>() / float(num_edge);

        for (int64_t e = 0; e < num_edge; ++e)
        {
            size_t i0 = edge2vtx[2 * e + 0];
            size_t i1 = edge2vtx[2 * e + 1];

            float ex = vtx_ptr[i1 * 2 + 0] - vtx_ptr[i0 * 2 + 0];
            float ey = vtx_ptr[i1 * 2 + 1] - vtx_ptr[i0 * 2 + 1];

            float len2 = ex * ex + ey * ey + 1e-8f;

            float phi = std::atan2(ey, ex);

            float tx = tdir_ptr[e * 2 + 0];
            float ty = tdir_ptr[e * 2 + 1];
            float theta = std::atan2(ty, tx);

            float d = phi - theta;
            if (d > M_PI)
                d -= 2.f * M_PI;
            if (d < -M_PI)
                d += 2.f * M_PI;

            // dL / dd
            float g = 4.f * std::sin(4.f * d) * gscale;

            // dφ / d(ex, ey)
            float dex = g * (-ey / len2);
            float dey = g * (ex / len2);

            gv[i0 * 2 + 0] -= dex;
            gv[i0 * 2 + 1] -= dey;
            gv[i1 * 2 + 0] += dex;
            gv[i1 * 2 + 1] += dey;
        }

        return {
            grad_vtx,        // vtx2xy
            torch::Tensor(), // edge2vtx (None)
            torch::Tensor()  // theta_dir (constant)
        };
    }

    torch::Tensor WallChainLossFunction::forward(
        torch::autograd::AutogradContext *ctx,
        const torch::Tensor &vtx2xy,
        const std::vector<std::vector<size_t>> &chains)
    {
        TORCH_CHECK(vtx2xy.is_cpu());
        TORCH_CHECK(vtx2xy.dtype() == torch::kFloat32);

        const float *vtx = vtx2xy.data_ptr<float>();

        std::vector<int64_t> edge_i0, edge_i1, chain_id;
        std::vector<int64_t> chain_offset;

        chain_offset.push_back(0);

        for (size_t c = 0; c < chains.size(); ++c)
        {
            const auto &chain = chains[c];
            if (chain.size() < 3)
                continue;

            for (size_t i = 0; i + 1 < chain.size(); ++i)
            {
                edge_i0.push_back(chain[i]);
                edge_i1.push_back(chain[i + 1]);
                chain_id.push_back(c);
            }
            chain_offset.push_back(edge_i0.size());
        }

        int64_t E = edge_i0.size();
        int64_t C = chain_offset.size() - 1;

        auto opts_i = torch::TensorOptions().dtype(torch::kInt64);
        auto opts_f = vtx2xy.options();

        torch::Tensor t_i0 = torch::from_blob(edge_i0.data(), {E}, opts_i).clone();
        torch::Tensor t_i1 = torch::from_blob(edge_i1.data(), {E}, opts_i).clone();
        torch::Tensor t_cid = torch::from_blob(chain_id.data(), {E}, opts_i).clone();
        torch::Tensor t_coff = torch::from_blob(chain_offset.data(), {C + 1}, opts_i).clone();

        torch::Tensor loss = torch::zeros({}, opts_f);

        for (int64_t c = 0; c < C; ++c)
        {
            int64_t beg = chain_offset[c];
            int64_t end = chain_offset[c + 1];

            float sx = 0.f, sy = 0.f;

            for (int64_t e = beg; e < end; ++e)
            {
                size_t i0 = edge_i0[e];
                size_t i1 = edge_i1[e];

                float ex = vtx[i1 * 2] - vtx[i0 * 2];
                float ey = vtx[i1 * 2 + 1] - vtx[i0 * 2 + 1];
                float len = std::sqrt(ex * ex + ey * ey) + 1e-8f;
                ex /= len;
                ey /= len;

                sx += ex;
                sy += ey;
            }

            float sl = std::sqrt(sx * sx + sy * sy) + 1e-8f;
            float nx = sx / sl, ny = sy / sl;

            for (int64_t e = beg; e < end; ++e)
            {
                size_t i0 = edge_i0[e];
                size_t i1 = edge_i1[e];
                float ex = vtx[i1 * 2] - vtx[i0 * 2];
                float ey = vtx[i1 * 2 + 1] - vtx[i0 * 2 + 1];
                float len = std::sqrt(ex * ex + ey * ey) + 1e-8f;
                ex /= len;
                ey /= len;

                float dot = ex * nx + ey * ny;
                loss += 1.f - dot * dot;
            }
        }

        ctx->save_for_backward({vtx2xy, t_i0, t_i1, t_coff});
        ctx->saved_data["num_edges"] = E;

        return loss / std::max<int64_t>(1, E);
    }

    torch::autograd::tensor_list WallChainLossFunction::backward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::tensor_list grad_outputs)
    {
        float g = grad_outputs[0].item<float>();

        auto saved = ctx->get_saved_variables();
        auto vtx2xy = saved[0];
        auto t_i0 = saved[1];
        auto t_i1 = saved[2];
        auto t_coff = saved[3];

        const float *vtx = vtx2xy.data_ptr<float>();
        const int64_t *i0 = t_i0.data_ptr<int64_t>();
        const int64_t *i1 = t_i1.data_ptr<int64_t>();
        const int64_t *coff = t_coff.data_ptr<int64_t>();

        int64_t C = t_coff.numel() - 1;
        int64_t E = ctx->saved_data["num_edges"].toInt();

        torch::Tensor grad_vtx = torch::zeros_like(vtx2xy);
        float *gv = grad_vtx.data_ptr<float>();

        float scale = g / std::max<int64_t>(1, E);

        for (int64_t c = 0; c < C; ++c)
        {
            int64_t beg = coff[c];
            int64_t end = coff[c + 1];

            float sx = 0.f, sy = 0.f;
            for (int64_t e = beg; e < end; ++e)
            {
                float ex = vtx[i1[e] * 2] - vtx[i0[e] * 2];
                float ey = vtx[i1[e] * 2 + 1] - vtx[i0[e] * 2 + 1];
                float len = std::sqrt(ex * ex + ey * ey) + 1e-8f;
                sx += ex / len;
                sy += ey / len;
            }

            float sl = std::sqrt(sx * sx + sy * sy) + 1e-8f;
            float nx = sx / sl, ny = sy / sl;

            for (int64_t e = beg; e < end; ++e)
            {
                size_t a = i0[e], b = i1[e];
                float ex = vtx[b * 2] - vtx[a * 2];
                float ey = vtx[b * 2 + 1] - vtx[a * 2 + 1];
                float len = std::sqrt(ex * ex + ey * ey) + 1e-8f;
                ex /= len;
                ey /= len;

                float dot = ex * nx + ey * ny;
                float gx = -2.f * dot * nx * scale;
                float gy = -2.f * dot * ny * scale;

                gv[a * 2] -= gx;
                gv[a * 2 + 1] -= gy;
                gv[b * 2] += gx;
                gv[b * 2 + 1] += gy;
            }
        }

        return {grad_vtx, torch::Tensor()};
    }

   

    static bool has_nan_inf(const torch::Tensor &t)
    {
        auto cont = t.contiguous().to(torch::kCPU);
        auto ptr = cont.data_ptr<float>();
        auto n = (size_t)cont.numel();
        for (size_t i = 0; i < n; ++i)
        {
            float v = ptr[i];
            if (!std::isfinite(v))
                return true;
        }
        return false;
    }

    static void print_tensor_info(const std::string &name, const torch::Tensor &t)
    {
        try
        {
            auto tc = t.contiguous().to(torch::kCPU);
            std::cout << name << " dtype=" << tc.dtype() << " shape=";
            for (auto s : tc.sizes())
                std::cout << s << "x";
            std::cout << "  numel=" << tc.numel();
            if (tc.numel() > 0 && tc.dtype() == torch::kFloat32)
            {
                const float *p = tc.data_ptr<float>();
                float mn = p[0], mx = p[0];
                for (size_t i = 0; i < tc.numel(); ++i)
                {
                    float v = p[i];
                    if (std::isnan(v))
                    {
                        std::cout << " NAN";
                        break;
                    }
                    if (v < mn)
                        mn = v;
                    if (v > mx)
                        mx = v;
                }
                std::cout << " min=" << mn << " max=" << mx;
                if (has_nan_inf(tc))
                    std::cout << " HAS_NAN_OR_INF";
            }
            std::cout << std::endl;
        }
        catch (const std::exception &e)
        {
            std::cerr << "print_tensor_info exception for " << name << ": " << e.what() << std::endl;
        }
    }

    void test_backward_cpp_exact(
        const std::vector<Vector2> &boundary,
        const std::vector<Vector2> &sites)
    {
        // ---------- 1. boundary polygon ----------
        std::vector<float> vtxl2xy = M2::flat_vec2(boundary);

        // ---------- 2. sites ----------
        std::vector<float> site_xy_vec = M2::flat_vec2(sites);
        const int64_t num_site = site_xy_vec.size() / 2;

        torch::Tensor site2xy0 = torch::from_blob(
                                     site_xy_vec.data(),
                                     {num_site, 2},
                                     torch::TensorOptions().dtype(torch::kFloat32))
                                     .clone()
                                     .set_requires_grad(true);

        // ---------- 3. Voronoi forward ----------
        auto [vtxv2xy0, vi, site2cell] = voronoi(vtxl2xy, site2xy0, [](size_t)
                                                 { return true; });

        // ---------- 4. random goal ----------
        torch::Tensor vtxv2xy_goal = torch::randn_like(vtxv2xy0);

        // ---------- 5. loss + backward ----------
        torch::Tensor loss = (vtxv2xy0 * vtxv2xy_goal).sum();
        loss.backward();

        torch::Tensor grad_site = site2xy0.grad().detach().clone();

        // ---------- 6. finite difference ----------
        const float eps = 1e-4f;
        auto grad_site_acc = grad_site.accessor<float, 2>();

        float max_abs_diff = 0.f;
        int bad_count = 0;

        for (int64_t i_site = 0; i_site < num_site; ++i_site)
        {
            for (int d = 0; d < 2; ++d)
            {
                torch::Tensor site_pert = site2xy0.detach().clone();
                auto pert_acc = site_pert.accessor<float, 2>();
                pert_acc[i_site][d] += eps;
                torch::Tensor vtxv2xy1 = VoronoiFunction::apply(site_pert, vtxl2xy, vi.vtxv2info);

                torch::Tensor loss1 = (vtxv2xy1 * vtxv2xy_goal).sum();

                float numeric = (loss1.item<float>() - loss.item<float>()) / eps;
                float analytic = grad_site_acc[i_site][d];

                float diff = numeric - analytic;
                float abs_diff = std::abs(diff);

                max_abs_diff = std::max(max_abs_diff, abs_diff);

                if (abs_diff > 5e-2f)
                    ++bad_count;

                std::cout
                    << "[site " << i_site
                    << " dim " << d << "] "
                    << "numeric=" << numeric
                    << " analytic=" << analytic
                    << " diff=" << diff
                    << std::endl;
            }
        }

        std::cout << "\n========== SUMMARY ==========\n";
        std::cout << "num_site        : " << num_site << "\n";
        std::cout << "eps             : " << eps << "\n";
        std::cout << "max_abs_diff    : " << max_abs_diff << "\n";
        std::cout << "bad_count (>0.05): " << bad_count << "\n";
        std::cout << "=============================\n";
    }
}
