#include "diffVoronoi.h"
#include <stdexcept>
#include <limits>
#include <iostream>

namespace diffVoronoi
{

    static constexpr int64_t SIZE_MAX_T = std::numeric_limits<int64_t>::max();

    // ---- Utility: convert float32 tensor row i to Eigen::Vector2d
    static Vector2d tensor_row_to_eigen_vec2(const float *ptr, int64_t i){
        return Vector2d(static_cast<double>(ptr[2 * i + 0]), static_cast<double>(ptr[2 * i + 1]));
    }

    // ---- Utility: convert vector<Vector2d> -> float32 tensor (N,2)
    static torch::Tensor eigen_vec_list_to_tensor_float32(const std::vector<Vector2d> &list)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor t = torch::zeros({static_cast<int64_t>(list.size()), 2}, opts);
        float *outp = t.data_ptr<float>();
        for (size_t i = 0; i < list.size(); ++i)
        {
            outp[2 * i + 0] = static_cast<float>(list[i].x());
            outp[2 * i + 1] = static_cast<float>(list[i].y());
        }
        return t;
    }

    // ---- VoronoiFunction::forward
    torch::Tensor VoronoiFunction::forward(torch::autograd::AutogradContext *ctx,
                                           const torch::Tensor &site2xy,
                                           const std::vector<double> &vtxl2xy,
                                           const std::vector<std::array<size_t, 4>> &vtxv2info){
        TORCH_CHECK(site2xy.device().is_cpu(), "site2xy must be on CPU");
        TORCH_CHECK(site2xy.dtype() == torch::kFloat32, "site2xy must be float32");
        TORCH_CHECK(site2xy.dim() == 2 && site2xy.size(1) == 2, "site2xy must be (N,2)");

        // Save site2xy for backward
        ctx->save_for_backward({site2xy});

        // Save params into saved_data as IValues
        ctx->saved_data["vtxl2xy"] = vtxl2xy; // IValue supports std::vector<double>
        // Convert vtxv2info (array<size_t,4>) to vector of vectors<size_t>
        std::vector<std::vector<size_t>> vtxv2info_vec;
        vtxv2info_vec.reserve(vtxv2info.size());
        for (const auto &a : vtxv2info){
            vtxv2info_vec.push_back({a[0], a[1], a[2], a[3]});
        }
        ctx->saved_data["vtxv2info"] = vtxv2info_vec;

        // Prepare site2xy as vector<double> flattened for calling voronoi2 functions
        const size_t num_site = site2xy.size(0);
        const float *site_ptr = site2xy.data_ptr<float>();
        std::vector<double> site_flat;
        site_flat.resize(num_site * 2);
        for (size_t i = 0; i < num_site; ++i){
            site_flat[2 * i + 0] = static_cast<double>(site_ptr[2 * i + 0]);
            site_flat[2 * i + 1] = static_cast<double>(site_ptr[2 * i + 1]);
        }
        // compute vtxv positions
        size_t num_vtxv = vtxv2info.size();
        std::vector<Vector2d> out;
        out.resize(num_vtxv, Vector2d::Zero());
        for (size_t i = 0; i < num_vtxv; ++i){
            out[i] = voronoi2::position_of_voronoi_vertex(vtxv2info[i],vtxl2xy,site_flat);
        }

        return eigen_vec_list_to_tensor_float32(out);
    }

    // ---- VoronoiFunction::backward
    torch::autograd::tensor_list VoronoiFunction::backward(torch::autograd::AutogradContext *ctx,
                                                 torch::autograd::tensor_list grad_outputs){
        // grad_outputs[0] is gradient wrt forward output vtxv2xy (M,2) float32
        TORCH_CHECK(grad_outputs.size() >= 1, "expected grad_outputs[0]");

        auto saved = ctx->get_saved_variables();
        TORCH_CHECK(saved.size() >= 1, "saved variables missing");
        torch::Tensor site2xy = saved[0];
        TORCH_CHECK(site2xy.device().is_cpu(), "site2xy must be CPU");
        TORCH_CHECK(site2xy.dtype() == torch::kFloat32, "site2xy must be float32");

        // recover params
        auto iv_vtxl2xy = ctx->saved_data["vtxl2xy"];
        auto iv_vtxv2info = ctx->saved_data["vtxv2info"];
        std::vector<double> vtxl2xy = iv_vtxl2xy.toDoubleVector();
        // convert vtxv2info back
        std::vector<std::array<int64_t, 4>> vtxv2info;
        {
            auto outer = iv_vtxv2info.toList();
            vtxv2info.reserve(outer.size());
            for (size_t i = 0; i < outer.size(); ++i)
            {
                auto inner = outer.get(i).toIntVector(); // returns vector<int64_t>
                std::array<int64_t, 4> arr = {inner[0], inner[1], inner[2], inner[3]};
                vtxv2info.push_back(arr);
            }
        }

        // load site2xy into vector<Vector2d>
        const int64_t num_site = site2xy.size(0);
        const float *site_ptr = site2xy.data_ptr<float>();
        std::vector<Vector2d> site_vec;
        site_vec.resize(static_cast<size_t>(num_site));
        for (int64_t i = 0; i < num_site; ++i)
            site_vec[i] = tensor_row_to_eigen_vec2(site_ptr, i);

        // load grad_outputs[0] into vector<Vector2d> dv
        torch::Tensor dv_tensor = grad_outputs[0];
        TORCH_CHECK(dv_tensor.dim() == 2 && dv_tensor.size(1) == 2, "grad output must be (M,2)");
        TORCH_CHECK(dv_tensor.dtype() == torch::kFloat32, "grad output must be float32");
        const int64_t num_vtxv = dv_tensor.size(0);
        const float *dv_ptr = dv_tensor.data_ptr<float>();
        std::vector<Vector2d> dv_vec;
        dv_vec.resize(static_cast<size_t>(num_vtxv));
        for (int64_t i = 0; i < num_vtxv; ++i)
            dv_vec[i] = tensor_row_to_eigen_vec2(dv_ptr, i);

        // accumulate dw_site2xy (double)
        std::vector<double> dw_site2xy_flat(static_cast<size_t>(num_site) * 2, 0.0);

        for (int64_t i_vtxv = 0; i_vtxv < num_vtxv; ++i_vtxv)
        {
            const auto &info = vtxv2info[static_cast<size_t>(i_vtxv)];
            // info are int64 (may be SIZE_MAX_T for absent)
            if (info[1] == static_cast<int64_t>(SIZE_MAX_T))
            {
                // original polygon vertex -> no grad w.r.t. sites
                continue;
            }
            else if (info[3] == static_cast<int64_t>(SIZE_MAX_T))
            {
                // intersection of loop edge and two sites
                size_t num_vtxl = vtxl2xy.size() / 2;
                size_t i1_loop = static_cast<size_t>(info[0]);
                if (!(i1_loop < num_vtxl))
                    throw std::runtime_error("loop index out of range");
                size_t i2_loop = (i1_loop + 1) % num_vtxl;

                Vector2d l1(vtxl2xy[2 * i1_loop], vtxl2xy[2 * i1_loop + 1]);
                Vector2d l2(vtxl2xy[2 * i2_loop], vtxl2xy[2 * i2_loop + 1]);

                size_t i0_site = static_cast<size_t>(info[1]);
                size_t i1_site = static_cast<size_t>(info[2]);
                Vector2d s0 = site_vec[i0_site];
                Vector2d s1 = site_vec[i1_site];

                // call utility: returns r, dr/ds0 (Matrix2d), dr/ds1 (Matrix2d)
                Vector2d r;
                Matrix2d drds0, drds1;
                std::tie(r, drds0, drds1) = util::dw_intersection_against_bisector(l1, (l2 - l1), s0, s1);

                Vector2d dv = dv_vec[static_cast<size_t>(i_vtxv)];

                Vector2d ds0 = drds0.transpose() * dv;
                Vector2d ds1 = drds1.transpose() * dv;

                dw_site2xy_flat[i0_site * 2 + 0] += ds0.x();
                dw_site2xy_flat[i0_site * 2 + 1] += ds0.y();
                dw_site2xy_flat[i1_site * 2 + 0] += ds1.x();
                dw_site2xy_flat[i1_site * 2 + 1] += ds1.y();
            }
            else
            {
                // circumcenter of three sites
                size_t idx0 = static_cast<size_t>(info[1]);
                size_t idx1 = static_cast<size_t>(info[2]);
                size_t idx2 = static_cast<size_t>(info[3]);

                Vector2d s0 = site_vec[idx0];
                Vector2d s1 = site_vec[idx1];
                Vector2d s2 = site_vec[idx2];

                util::CircumcenterResult circumResult = util::wdw_circumcenter(s0, s1, s2);
                std::array<Matrix2d, 3> dvds = circumResult.dcc;

                Vector2d dv = dv_vec[static_cast<size_t>(i_vtxv)];
                for (int i_node = 0; i_node < 3; ++i_node)
                {
                    Vector2d ds = dvds[i_node].transpose() * dv;
                    size_t is0 = (i_node == 0 ? idx0 : (i_node == 1 ? idx1 : idx2));
                    dw_site2xy_flat[is0 * 2 + 0] += ds.x();
                    dw_site2xy_flat[is0 * 2 + 1] += ds.y();
                }
            }
        }

        // convert dw_site2xy_flat -> torch tensor float32 same shape as site2xy
        auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        torch::Tensor grad_site = torch::zeros({num_site, 2}, opts);
        float *gptr = grad_site.data_ptr<float>();
        for (int64_t i = 0; i < num_site; ++i)
        {
            gptr[2 * i + 0] = static_cast<float>(dw_site2xy_flat[2 * i + 0]);
            gptr[2 * i + 1] = static_cast<float>(dw_site2xy_flat[2 * i + 1]);
        }

        // return gradient for first arg (site2xy), and nullptrs for other (vtxl2xy/vtxv2info) which are constants
        return {grad_site, torch::Tensor(), torch::Tensor()};
    }

    // ---- VoronoiLayer wrapper implementation
    VoronoiLayer::VoronoiLayer(const std::vector<double> &vtxl2xy_in,
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

}
