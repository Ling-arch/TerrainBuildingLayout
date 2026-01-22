#pragma once

#include "voronoi2.h"
#include "diffVoronoi.h"
#include "loss.h"
#include "polygonMesh.h"
#include "polyloop.h"
#include <vector>
#include "tensorfield.h"

namespace optimizer
{
    using std::vector, std::tuple,std::pair;
    using Scalar = voronoi2::Scalar;
    using M2 = util::Math2<Scalar>;
    using Vector2 = typename M2::Vector2;
    using Vector3 = typename M2::Vector3;
    using Mat2 = typename M2::Matrix2;
    using polyloop::Polyloop2, polyloop::Polyloop3;

    inline constexpr size_t INVALID = static_cast<size_t>(std::numeric_limits<int64_t>::max());

    //生成点时，根据平面跨层情况，选择polyloop的相交区域进行生成，同时记录当前区域的站点id，后续分配房间
    //优先考虑相交区域，将跨层的房间id先分配给
    struct MultiPlanProblem{
        vector<vector<Scalar>> vtxl2xy_norms;
    };
    
    struct PlanProblem
    {
        //归一化后的plan
        vector<Scalar> vtxl2xy_norm;
        polyloop::NormalizeTransform2D<Scalar> tf;
        field::TensorField2D<Scalar> field;
        int64_t num_room;
        int floor;

        // room area target
        vector<Scalar> room2area_trg;

        // site_id to room_id
        vector<size_t> site2room;

        //room typology
        vector<pair<size_t, size_t>> room_connections;

        // fixed flags
        vector<Scalar> site2xy2flag;

        // sites
        vector<Vector2> sites_norm;

        // constants
        Scalar total_area_trg;
        std::unique_ptr<torch::optim::AdamW> optimizer;
        torch::TensorOptions options;
        torch::Tensor site2xy;
        torch::Tensor site2xy_ini;
        torch::Tensor site2xy2flag_t;
        torch::Tensor room2area_trg_t;
    };


    struct OptimizeDrawData{
        vector<Polyloop3> cellPolys;
        vector<Vector3> sites_world;
        vector<vector<Vector3>> wall_edge_list;
        vector<size_t> site_ids;
    };

    PlanProblem define_problem(const int floor, const vector<Vector2> &boundary, const vector<float> &area_ratio, const vector<pair<size_t, size_t>> &room_connections, const vector<Vector2> &fix_sites, const vector<size_t> &fix_rooms);
    PlanProblem define_field_problem(const int floor,field::TensorField2D<float> field, const vector<Vector2> &boundary, const vector<float> &area_ratio, const vector<pair<size_t, size_t>> &room_connections, const vector<Vector2> &fix_sites, const vector<size_t> &fix_rooms);
    void optimize_draw_bystep(PlanProblem& plan_prob, size_t &cur_iter, const size_t max_iter, OptimizeDrawData &diff_voronoi_show_data);
    void optimize_field_problem_and_draw_bystep(PlanProblem &plan_prob, size_t &cur_iter, const size_t max_iter, OptimizeDrawData &diff_voronoi_show_data);
}