#include "voronoi2.h"


namespace voronoi2
{

    bool Cell::is_inside(const Vector2 &p) const
    {
        Scalar wn = M2::winding_number(vtx2xy, p);
        return std::abs(wn - Scalar(1)) < Scalar(0.1);
    };

    Scalar Cell::area() const
    {
        return M2::polygon_area(vtx2xy);
    };

    Cell new_from_polyloop2(const vector<Vector2> &vtx2xy_in)
    {
        Cell c;
        c.vtx2xy = vtx2xy_in;
        c.vtx2info.resize(vtx2xy_in.size());
        for (size_t i = 0; i < vtx2xy_in.size(); ++i)
        {
            c.vtx2info[i] = {i, INVALID_INDEX, INVALID_INDEX, INVALID_INDEX};
        }
        return c;
    };

    Cell new_empty()
    {
        Cell c;
        c.vtx2xy.clear();
        c.vtx2info.clear();
        return c;
    };

    std::optional<Cell> hoge(
        const std::vector<Vector2> &vtx2xy,
        const std::vector<std::array<size_t, 4>> &vtx2info,
        const std::vector<std::tuple<float, size_t, Vector2, std::array<size_t, 4>>> &vtxnews,
        const std::vector<size_t> &vtx2vtxnew,
        std::vector<bool> &vtxnew2isvisited)
    {
        const size_t N = vtx2xy.size();
        const size_t M = vtxnews.size();

        std::vector<Vector2> vtx2xy_new;
        std::vector<std::array<size_t, 4>> vtx2info_new;

        // 1. find first unvisited new vertex
        size_t i_vtx0 = INVALID_INDEX;
        for (size_t i = 0; i < M; ++i)
        {
            if (!vtxnew2isvisited[i])
            {
                i_vtx0 = i;
                break;
            }
        }
        if (i_vtx0 == INVALID_INDEX)
            return std::nullopt;

        bool is_new0 = true;
        size_t i_vtx = i_vtx0;
        bool is_new = true;
        bool is_entry = true;

        while (true)
        {
            if (is_new)
            {
                // --- new vertex ---
                assert(i_vtx < M);

                vtx2xy_new.push_back(std::get<2>(vtxnews[i_vtx]));
                vtx2info_new.push_back(std::get<3>(vtxnews[i_vtx]));
                vtxnew2isvisited[i_vtx] = true;

                if (is_entry)
                {
                    // jump into original polygon
                    size_t orig = std::get<1>(vtxnews[i_vtx]);
                    assert(orig < N);
                    i_vtx = (orig + 1) % N;
                    is_new = false;
                    is_entry = false;
                }
                else
                {
                    // move backward in original polygon (with wrap)
                    i_vtx = (i_vtx + N - 1) % N;
                    is_new = true;
                    is_entry = true;
                }
            }
            else
            {
                // --- original vertex ---
                assert(i_vtx < N);

                vtx2xy_new.push_back(vtx2xy[i_vtx]);
                vtx2info_new.push_back(vtx2info[i_vtx]);

                size_t j = vtx2vtxnew[i_vtx];
                if (j == INVALID_INDEX)
                {
                    i_vtx = (i_vtx + 1) % N;
                    is_new = false;
                }
                else
                {
                    assert(j < M);
                    i_vtx = j;
                    is_new = true;
                    is_entry = false;
                }
            }

            // termination
            if (i_vtx == i_vtx0 && is_new == is_new0)
                break;
        }

        return Cell{vtx2xy_new, vtx2info_new};
    }

    std::vector<Cell> cut_polygon_by_line(
        const Cell &cell,
        const Vector2 &line_s,
        const Vector2 &line_n,
        size_t i_vtx,
        size_t j_vtx)
    {
        auto depth = [&](const Vector2 &p)
        {
            return (p - line_s).dot(line_n);
        };

        Vector2 line_t = M2::rotate90(line_n);

        bool is_inside = false;
        std::vector<std::tuple<Scalar, size_t, Vector2, std::array<size_t, 4>>> vtxnews;
        vtxnews.reserve(cell.vtx2xy.size());

        constexpr Scalar EPS = Scalar(1e-12);

        for (size_t i0 = 0; i0 < cell.vtx2xy.size(); ++i0)
        {
            size_t i1 = (i0 + 1) % cell.vtx2xy.size();

            const Vector2 &p0 = cell.vtx2xy[i0];
            const Vector2 &p1 = cell.vtx2xy[i1];

            Scalar d0 = depth(p0);
            Scalar d1 = depth(p1);

            if (d0 < 0)
                is_inside = true;

            if (std::abs(d0) < EPS && std::abs(d1) < EPS)
                continue;
            if (d0 * d1 > 0)
                continue;

            Vector2 pm = p0 * (d1 / (d1 - d0)) + p1 * (d0 / (d0 - d1));
            Scalar t0 = line_t.dot(pm);

            const auto &info0 = cell.vtx2info[i0];
            const auto &info1 = cell.vtx2info[i1];

            std::array<size_t, 4> info;
            std::vector<size_t> inter;

            for (size_t a : {info0[2], info0[3]})
                if (a != INVALID_INDEX &&
                    (a == info1[2] || a == info1[3]))
                    inter.push_back(a);

            if (inter.empty())
                info = {info0[0], i_vtx, j_vtx, INVALID_INDEX};
            else if (inter.size() == 1)
                info = {INVALID_INDEX, i_vtx, inter[0], j_vtx};
            else
                assert(false);

            vtxnews.emplace_back(-t0, i0, pm, info);
        }

        if (vtxnews.empty())
            return is_inside ? std::vector<Cell>{cell} : std::vector<Cell>{};

        std::sort(vtxnews.begin(), vtxnews.end(),
                  [](auto &a, auto &b)
                  { return std::get<0>(a) < std::get<0>(b); });

        assert(vtxnews.size() % 2 == 0);

        std::vector<size_t> vtx2vtxnew(cell.vtx2xy.size(), INVALID_INDEX);
        for (size_t i = 0; i < vtxnews.size(); ++i)
        {
            size_t orig = std::get<1>(vtxnews[i]);
            assert(vtx2vtxnew[orig] == INVALID_INDEX);
            vtx2vtxnew[orig] = i;
        }

        std::vector<bool> visited(vtxnews.size(), false);
        std::vector<Cell> cells;

        while (true)
        {
            auto opt = hoge(cell.vtx2xy, cell.vtx2info, vtxnews, vtx2vtxnew, visited);
            if (!opt)
                break;
            cells.push_back(std::move(*opt));
        }

        return cells;
    }

    vector<Cell> voronoi_cells(
        const vector<Scalar> &vtxl2xy_flat, // flattened [x0,y0, x1,y1, ...] outer loop
        const vector<Scalar> &site2xy_flat, // flattened sites [x0,y0, ...]
        const std::function<bool(size_t)> &site2isalive)
    {
        vector<Vector2> vtxl2xy = M2::to_vec2_array(vtxl2xy_flat);
        vector<Vector2> site2xy = M2::to_vec2_array(site2xy_flat);

        size_t num_site = site2xy.size();
        vector<Cell> site2cell(num_site, new_empty());

        for (size_t i_site = 0; i_site < num_site; ++i_site)
        {
            if (!site2isalive(i_site))
                continue;
            vector<Cell> cell_stack;
            cell_stack.push_back(new_from_polyloop2(vtxl2xy));
            for (size_t j_site = 0; j_site < num_site; ++j_site)
            {
                if (!site2isalive(j_site))
                    continue;
                if (j_site == i_site)
                    continue;

                Vector2 line_s = Scalar(0.5) * (site2xy[i_site] + site2xy[j_site]);
                Vector2 line_n = (site2xy[j_site] - site2xy[i_site]).normalized();

                vector<Cell> cell_stack_new;
                for (auto &cell_in : cell_stack)
                {
                    auto parts = cut_polygon_by_line(cell_in, line_s, line_n, i_site, j_site);
                    cell_stack_new.insert(cell_stack_new.end(), parts.begin(), parts.end());
                }
                cell_stack.swap(cell_stack_new);
                if (cell_stack.empty())
                    break;
            }

            if (cell_stack.empty())
            {
                site2cell[i_site] = new_empty();
                continue;
            }
            if (cell_stack.size() == 1)
            {
                site2cell[i_site] = cell_stack[0];
                continue;
            }
            // choose best by membership or area heuristic (like Rust)
            vector<std::pair<Scalar, size_t>> depthcell;
            for (size_t k = 0; k < cell_stack.size(); ++k)
            {
                bool inside = cell_stack[k].is_inside(site2xy[i_site]);
                Scalar score = inside ? 0.0 : 1.0 / std::max(Scalar(1e-12), cell_stack[k].area());
                depthcell.emplace_back(score, k);
            }
            std::sort(depthcell.begin(), depthcell.end(),
                      [](auto &a, auto &b)
                      { return a.first < b.first; });
            size_t pick = depthcell[0].second;
            site2cell[i_site] = cell_stack[pick];
        }

        return site2cell;
    }

    // 索引化：把每个 cell 的局部顶点去重映射到全局 voronoi 顶点
    VoronoiMesh indexing(const vector<Cell> &site2cell)
    {
        size_t num_site = site2cell.size();

        auto sort_info = [](const array<size_t, 4> &info)
        {
            array<size_t, 4> tmp = info;
            array<size_t, 3> t = {tmp[1], tmp[2], tmp[3]};
            std::sort(t.begin(), t.end());
            return array<size_t, 4>{tmp[0], t[0], t[1], t[2]};
        };

        // mapping info -> global voronoi vertex index
        std::map<array<size_t, 4>, size_t> info2vtxv;
        vector<array<size_t, 4>> vtxv2info;

        for (const auto &cell : site2cell)
        {
            for (const auto &info : cell.vtx2info)
            {
                auto info0 = sort_info(info);
                if (!info2vtxv.count(info0))
                {
                    size_t id = info2vtxv.size();
                    info2vtxv[info0] = id;
                    vtxv2info.push_back(info0);
                }
            }
        }

        size_t num_vtxv = info2vtxv.size();
        vector<Vector2> vtxv2xy(num_vtxv, Vector2::Zero());
        vector<size_t> site2idx;
        site2idx.reserve(num_site + 1);
        site2idx.push_back(0);
        vector<size_t> idx2vtxc;
        idx2vtxc.reserve(num_vtxv * 2);

        for (const auto &cell : site2cell)
        {
            for (size_t ind = 0; ind < cell.vtx2info.size(); ++ind)
            {
                auto info0 = sort_info(cell.vtx2info[ind]);
                size_t i_vtxv = info2vtxv.at(info0);
                idx2vtxc.push_back(i_vtxv);
                vtxv2xy[i_vtxv] = cell.vtx2xy[ind];
            }
            site2idx.push_back(idx2vtxc.size());
        }

        VoronoiMesh vm;
        vm.site2idx = std::move(site2idx);
        vm.idx2vtxv = std::move(idx2vtxc);
        vm.vtxv2xy = std::move(vtxv2xy);
        vm.vtxv2info = std::move(vtxv2info);
        return vm;
    }

    Vector2 position_of_voronoi_vertex(const array<size_t, 4> &info, const vector<Scalar> &vtxl2xy, const vector<Scalar> &site2xy)
    {
        // helpers to fetch coordinates
        auto get_vtxl = [&](size_t idx) -> Vector2
        {
            size_t off = idx * 2;
            return Vector2(vtxl2xy[off], vtxl2xy[off + 1]);
        };
        auto get_site = [&](size_t idx) -> Vector2
        {
            size_t off = idx * 2;
            return Vector2(site2xy[off], site2xy[off + 1]);
        };

        if (info[1] == INVALID_INDEX)
        {
            // original loop vertex
            //std::cout << "original vertex " << get_vtxl(info[0]) <<std::endl;
            return get_vtxl(info[0]);
        }
        else if (info[3] == INVALID_INDEX)
        {
            
            // intersection of loop edge and bisector between two sites
            size_t num_vtxl = vtxl2xy.size() / 2;
            assert(info[0] < num_vtxl);
            size_t i1_loop = info[0];
            size_t i2_loop = (i1_loop + 1) % num_vtxl;
            Vector2 l1 = get_vtxl(i1_loop);
            Vector2 l2 = get_vtxl(i2_loop);
            Vector2 s1 = get_site(info[1]);
            Vector2 s2 = get_site(info[2]);
            Vector2 mid = Scalar(0.5) * (s1 + s2);
            Vector2 bisdir = M2::rotate90(s2 - s1);
            //std::cout << "two sites intersect vertex , sites is" << info[1] << " , " << info[2] << std::endl;
            return M2::line_intersection(l1, (l2 - l1), mid, bisdir);
        }
        else
        {
            // three-site circumcenter
            assert(info[0] == INVALID_INDEX);
            Vector2 s0 = get_site(info[1]);
            Vector2 s1 = get_site(info[2]);
            Vector2 s2 = get_site(info[3]);
            //std::cout << "three sites intersect vertex , sites is" << info[1] << " , " << info[2] << " , " << info[3]<< std::endl;
            return M2::circumcenter(s0, s1, s2);
        }
    }

}