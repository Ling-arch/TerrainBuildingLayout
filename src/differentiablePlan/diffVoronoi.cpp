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
                std::cout << "original polygon vertex index is " << i_vtxv << std::endl;
                continue;
            }
            else if (info[3] == SIZE_MAX_T)
            {
                std::cout << "polygon edge with two sites vertex index is " << i_vtxv << std::endl;
                // intersection of loop edge and two sites
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
                std::cout << "Two sites is (" << i0_site <<"," << i1_site << ")"<< std::endl;
                // call utility: returns r, dr/ds0 (Matrix2d), dr/ds1 (Matrix2d)
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
                std::cout << "three sites vtxv index is " << i_vtxv << std::endl;
                // circumcenter of three sites
                size_t idx0 = info[1];
                size_t idx1 = info[2];
                size_t idx2 = info[3];
                std::cout << "Three sites is (" << idx0 << ", " << idx1 << ", " << idx2 << ")" << std::endl;
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

    // ---- VoronoiLayer wrapper implementation
    VoronoiLayer::VoronoiLayer(const std::vector<float> &vtxl2xy_in,
                               const std::vector<std::array<size_t, 4>> &vtxv2info_in)
    {
        vtxl2xy = vtxl2xy_in;
        vtxv2info = vtxv2info_in;
    }

    std::vector<std::array<size_t, 4>> VoronoiLayer::get_vtxv2info_i64() const
    {
        return vtxv2info;
    }

    torch::Tensor VoronoiLayer::forward(const torch::Tensor &site2xy) const
    {
        // call autograd::Function apply
        // note: torch::autograd::Function::apply signature expects the same types we defined for forward
        return VoronoiFunction::apply(site2xy, vtxl2xy, vtxv2info);
    }

    std::pair<torch::Tensor, VoronoiInfo>
    voronoi(
        const std::vector<float> &vtxl2xy_f,
        const torch::Tensor &site2xy,
        const std::function<bool(size_t)> &site2isalive)
    {
        // 1) flatten site tensor
        std::vector<float> site_flat = flat_tensor_to_float(site2xy);

        // 2) compute site2cell
        auto site2cell =
            voronoi2::voronoi_cells(vtxl2xy_f, site_flat, site2isalive);

        // 3) indexing
        voronoi2::VoronoiMesh voronoi_mesh =
            voronoi2::indexing(site2cell);

        // 4) autograd layer
        VoronoiLayer layer(
            std::vector<float>(vtxl2xy_f.begin(), vtxl2xy_f.end()),
            voronoi_mesh.vtxv2info);

        torch::Tensor vtxv2xy = layer.forward(site2xy);

        // --------------------------------------------------
        // ★★★ Rust 对齐点：idx2site（elem2elem）
        // --------------------------------------------------
        size_t num_vtxv = vtxv2xy.size(0);

        std::vector<size_t> idx2site =
            polygonMesh::elem2elem_from_polygon_mesh(
                voronoi_mesh.site2idx,
                voronoi_mesh.idx2vtxv,
                num_vtxv);

        // 5) pack VoronoiInfo
        VoronoiInfo vi;
        vi.site2idx = voronoi_mesh.site2idx;
        vi.idx2vtxv = voronoi_mesh.idx2vtxv;
        vi.vtxv2info = voronoi_mesh.vtxv2info;
        vi.idx2site = std::move(idx2site);

        return {vtxv2xy, vi};
    }

    // Forward: elem2idx (cumulative), idx2vtx (vertex indices), vtx2xy (num_vtx x 2 float tensor)
    // returns Tensor elem2cog (num_elem x 2 float)
    torch::Tensor polygonmesh2_to_cogs_forward(
        const std::vector<size_t> &elem2idx, // size = num_elem + 1
        const std::vector<size_t> &idx2vtx,  // flattened vertex index list
        const torch::Tensor &vtx2xy          // (num_vtx,2) float32
    )
    {
        TORCH_CHECK(vtx2xy.device().is_cpu(), "vtx2xy must be CPU tensor");
        TORCH_CHECK(vtx2xy.dim() == 2 && vtx2xy.size(1) == 2, "vtx2xy must be (N,2)");

        size_t num_elem = elem2idx.size() > 1 ? elem2idx.size() - 1 : 0;

        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor elem2cog = torch::zeros({(long)num_elem, 2}, options);

        // contiguous float* pointer
        auto vcont = vtx2xy.contiguous();
        const float *vptr = vcont.data_ptr<float>(); // layout: [x0,y0, x1,y1, ...]

        // accessor for output
        auto acc = elem2cog.accessor<float, 2>();

        // main loop
        for (size_t ie = 0; ie < num_elem; ++ie)
        {
            size_t start = elem2idx[ie];
            size_t end = elem2idx[ie + 1];
            size_t nv = (end > start) ? (end - start) : 0;

            if (nv == 0)
            {
                acc[ie][0] = 0.0f;
                acc[ie][1] = 0.0f;
                continue;
            }

            double sx = 0.0;
            double sy = 0.0;

            for (size_t k = start; k < end; ++k)
            {
                size_t vidx = idx2vtx[k];

                // direct pointer access
                // vtx2xy is (num_vtx,2)
                sx += vptr[vidx * 2 + 0];
                sy += vptr[vidx * 2 + 1];
            }

            float rx = float(sx / double(nv));
            float ry = float(sy / double(nv));

            // write using accessor
            acc[ie][0] = rx;
            acc[ie][1] = ry;
        }

        return elem2cog;
    }

    // Backward: distribute dw_elem2cog (num_elem x 2) back to vertices (num_vtx x 2)
    // returns dw_vtx2xy tensor (num_vtx x 2)
    torch::Tensor polygonmesh2_to_cogs_backward(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const torch::Tensor &vtx2xy,
        const torch::Tensor &dw_elem2cog)
    {
        TORCH_CHECK(dw_elem2cog.device().is_cpu());
        TORCH_CHECK(dw_elem2cog.dim() == 2 && dw_elem2cog.size(1) == 2);

        size_t num_vtx = (size_t)vtx2xy.size(0);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor dw_vtx2xy = torch::zeros({(long)num_vtx, 2}, options);

        auto acc = dw_vtx2xy.accessor<float, 2>();

        auto gcont = dw_elem2cog.contiguous();
        const float *gptr = gcont.data_ptr<float>();

        size_t num_elem = elem2idx.size() > 1 ? elem2idx.size() - 1 : 0;

        for (size_t ie = 0; ie < num_elem; ++ie)
        {
            size_t start = elem2idx[ie];
            size_t end = elem2idx[ie + 1];
            size_t nv = (end > start) ? (end - start) : 0;

            if (nv == 0)
                continue;

            float ratio = 1.0f / float(nv);
            float gx = gptr[ie * 2 + 0];
            float gy = gptr[ie * 2 + 1];

            for (size_t k = start; k < end; ++k)
            {
                size_t vidx = idx2vtx[k]; // vertex index

                // Just use accessor — fastest and cleanest
                acc[vidx][0] += ratio * gx;
                acc[vidx][1] += ratio * gy;
            }
        }

        return dw_vtx2xy;
    }

    torch::Tensor loss_lloyd(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const torch::Tensor &site2xy, // (num_sites,2) float32
        const torch::Tensor &vtxv2xy  // (num_vtxv,2) float32
    )
    {
        // compute element centers (site2cogs) from voronoi vertices (vtxv2xy) using polygonmesh2_to_cogs_forward
        torch::Tensor site2cogs = polygonmesh2_to_cogs_forward(elem2idx, idx2vtx, vtxv2xy);

        // compute difference site2xy - site2cogs
        TORCH_CHECK(site2xy.device().is_cpu() && site2cogs.device().is_cpu(), "tensors must be CPU");
        auto diff = site2xy - site2cogs; // broadcasting shapes must match
        auto sq = diff * diff;
        // sum all returns scalar tensor
        torch::Tensor loss = sq.sum();
        return loss;
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

        torch::Tensor site2xy0 =
            torch::from_blob(
                site_xy_vec.data(),
                {num_site, 2},
                torch::TensorOptions().dtype(torch::kFloat32))
                .clone()
                .set_requires_grad(true);

        // ---------- 3. Voronoi forward ----------
        auto [vtxv2xy0, vi] = voronoi(vtxl2xy, site2xy0, [](size_t){ return true; });

        // ---------- 4. random goal ----------
        torch::Tensor vtxv2xy_goal = torch::randn_like(vtxv2xy0);

        // ---------- 5. loss + backward ----------
        torch::Tensor loss = (vtxv2xy0 * vtxv2xy_goal).sum();
        loss.backward();

        torch::Tensor grad_site = site2xy0.grad().detach().clone();

        // ---------- 6. finite difference ----------
        const float eps = 1e-3f;
        auto grad_site_acc = grad_site.accessor<float, 2>();

        float max_abs_diff = 0.f;
        int bad_count = 0;

        for (int64_t i_site = 0; i_site < num_site; ++i_site)
        {
            for (int d = 0; d < 2; ++d)
            {
                torch::Tensor site_pert =site2xy0.detach().clone();
                auto pert_acc =site_pert.accessor<float, 2>();
                pert_acc[i_site][d] += eps;
                torch::Tensor vtxv2xy1 =VoronoiFunction::apply(site_pert,vtxl2xy,vi.vtxv2info);

                torch::Tensor loss1 =(vtxv2xy1 * vtxv2xy_goal).sum();

                float numeric =(loss1.item<float>() - loss.item<float>()) / eps;
                float analytic =grad_site_acc[i_site][d];

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
