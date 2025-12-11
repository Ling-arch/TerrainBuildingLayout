#include "voronoi2.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <iterator>

namespace voronoi2
{
    vector<Vector2d> to_vec2_array(const vector<double> &flat)
    {
        assert(flat.size() % 2 == 0);
        vector<Vector2d> out;
        out.reserve(flat.size() / 2);
        for (size_t i = 0; i + 1 < flat.size(); i += 2)
            out.emplace_back(flat[i], flat[i + 1]);
        return out;
    }

    // Solve intersection of two parametric lines:
    // L1: ls + t * ld; L2: qs + u * qd
    // returns point on L1 (ls + t*ld). If parallel, returns ls.
    Vector2d line_intersection_param(
        const Vector2d &ls, const Vector2d &ld,
        const Vector2d &qs, const Vector2d &qd)
    {
        // Solve: ls + t ld = qs + u qd  =>  t ld - u qd = qs - ls
        // 2x2 linear system for (t, u). A * [t; u] = rhs
        Matrix2d A;
        A.col(0) = ld;
        A.col(1) = -qd;
        double det = A.determinant();
        if (std::abs(det) < 1e-12)
        {
            // parallel-ish: fallback
            return ls;
        }
        Vector2d rhs = qs - ls;
        Vector2d sol = A.colPivHouseholderQr().solve(rhs);
        double t = sol(0);
        return ls + t * ld;
    }

    Vector2d circumcenter(const Vector2d &a, const Vector2d &b, const Vector2d &c)
    {
        // Standard circumcenter via perpendicular bisector intersection
        Vector2d mid_ab = 0.5 * (a + b);
        Vector2d dir_ab = b - a;
        Vector2d n1 = util::rotate90(dir_ab);

        Vector2d mid_bc = 0.5 * (b + c);
        Vector2d dir_bc = c - b;
        Vector2d n2 = util::rotate90(dir_bc);

        return line_intersection_param(mid_ab, n1, mid_bc, n2);
    }


    double polygon_area(const vector<Vector2d> &poly){
        if (poly.size() < 3)
            return 0.0;
        double s = 0.0;
        for (size_t i = 0; i < poly.size(); ++i)
        {
            size_t j = (i + 1) % poly.size();
            s += poly[i].x() * poly[j].y() - poly[j].x() * poly[i].y();
        }
        return 0.5 * std::abs(s);
    }

    // Winding number (robust-ish) for point-in-polygon
    double winding_number(const vector<Vector2d> &poly, const Vector2d &p){
        int wn = 0;

        for (size_t i = 0; i < poly.size(); ++i){
            Vector2d a = poly[i];
            Vector2d b = poly[(i + 1) % poly.size()];

            Vector2d ab = b - a;
            Vector2d ap = p - a;
            double isLeft = ab.x() * ap.y() - ab.y() * ap.x();
            if (a.y() <= p.y()){
                if (b.y() > p.y()){
                    if (isLeft > 0)
                        ++wn;
                }
            }
            else{
                if (b.y() <= p.y()){
                    if (isLeft < 0)
                        --wn;
                }
            }
        }

        return static_cast<double>(wn);
    }

    // ----------------- Cell 方法 -----------------
    Cell Cell::new_from_polyloop2(const vector<Vector2d> &vtx2xy_in){
        Cell c;
        c.vtx2xy = vtx2xy_in;
        c.vtx2info.resize(vtx2xy_in.size());
        for (size_t i = 0; i < vtx2xy_in.size(); ++i){
            c.vtx2info[i] = {i, std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max(), std::numeric_limits<size_t>::max()};
        }
        return c;
    }


    Cell Cell::new_empty(){
        return Cell{};
    }

    bool Cell::is_inside(const Vector2d &p) const{
        double wn = winding_number(vtx2xy, p);
        return std::abs(wn - 1.0) < 0.1;
    }

    double Cell::area() const{
        return polygon_area(vtx2xy);
    }

    // ----------------- hoge (polygon reconstruction helper) -----------------
    // This reconstructs one polygon from original vertices and newly inserted intersections.
    // It follows the control-flow of the Rust `hoge` function.
    static std::optional<Cell> hoge(
        const vector<Vector2d> &vtx2xy,
        const vector<array<size_t, 4>> &vtx2info,
        const vector<std::tuple<double, size_t, Vector2d, array<size_t, 4>>> &vtxnews,
        const vector<size_t> &vtx2vtxnew,
        vector<char> &vtxnew2isvisited){
        // find first not visited new vertex
        auto it = std::find(vtxnew2isvisited.begin(), vtxnew2isvisited.end(), 0);
        if (it == vtxnew2isvisited.end())
            return std::nullopt;
        size_t i_vtx0 = std::distance(vtxnew2isvisited.begin(), it);
        size_t i_vtx = i_vtx0;
        bool is_new = true;
        bool is_entry = true;

        vector<Vector2d> vtx2xy_new;
        vector<array<size_t, 4>> vtx2info_new;

        while (true){
            if (is_new){
                // take from vtxnews
                const auto &tpl = vtxnews[i_vtx];
                Vector2d pm = std::get<2>(tpl);
                array<size_t, 4> info = std::get<3>(tpl);
                vtx2xy_new.push_back(pm);
                vtx2info_new.push_back(info);
                vtxnew2isvisited[i_vtx] = 1;
                if (is_entry){
                    // jump to neighbor original vertex index stored in tuple.second
                    size_t orig = std::get<1>(tpl);
                    // in Rust they set i_vtx = vtxnews[i_vtx].1; then i_vtx = (i_vtx + 1) % vtx2xy.len();
                    i_vtx = (orig + 1) % vtx2xy.size();
                    is_new = false;
                    is_entry = false;
                }
                else{
                    // go backward one vertex on original polygon
                    if (i_vtx == 0)
                        i_vtx = vtx2xy.size() - 1;
                    else
                        i_vtx = i_vtx - 1;
                    is_new = true;
                    is_entry = true;
                }
            }
            else{
                // take from original polygon
                vtx2xy_new.push_back(vtx2xy[i_vtx]);
                vtx2info_new.push_back(vtx2info[i_vtx]);
                size_t mapped = vtx2vtxnew[i_vtx];
                if (mapped == std::numeric_limits<size_t>::max()){
                    // advance original
                    i_vtx = (i_vtx + 1) % vtx2xy.size();
                    is_new = false;
                }
                else{
                    i_vtx = mapped;
                    is_new = true;
                    is_entry = false;
                }
            }
            if (i_vtx == i_vtx0 && is_new == true && is_entry == true){
                // break when back to start with same state
                break;
            }
            // The original rust condition: if i_vtx==i_vtx0 && is_new==is_new0 { break; }
            // We started with is_new0=true implicitly.
            if (vtx2xy_new.size() > vtx2xy.size() + vtxnews.size() + 10){
                // safety to avoid infinite loop in malformed cases
                break;
            }
        }

        Cell cell;
        cell.vtx2xy = std::move(vtx2xy_new);
        cell.vtx2info = std::move(vtx2info_new);
        return cell;
    }

    // ----------------- cut_polygon_by_line -----------------
    // Mirrors the Rust logic: find intersections, construct vtxnews, then reconstruct pieces via hoge()
    vector<Cell> cut_polygon_by_line(
        const Cell &cell,
        const Vector2d &line_s,
        const Vector2d &line_n,
        size_t i_vtx,
        size_t j_vtx){
        auto depth = [&](const Vector2d &p){
            return (p - line_s).dot(line_n);
        };

        // line tangent / direction orthogonal to line_n
        Vector2d line_t = util::rotate90(line_n);

        bool is_inside = false;
        vector<std::tuple<double, size_t, Vector2d, array<size_t, 4>>> vtxnews;
        vtxnews.reserve(cell.vtx2xy.size());

        for (size_t i0 = 0; i0 < cell.vtx2xy.size(); ++i0){
            size_t i1 = (i0 + 1) % cell.vtx2xy.size();
            const Vector2d &p0 = cell.vtx2xy[i0];
            const Vector2d &p1 = cell.vtx2xy[i1];
            double d0 = depth(p0);
            if (d0 < 0.0)
                is_inside = true;
            double d1 = depth(p1);
            // avoid degenerate exactly zero product
            if (std::abs(d0) < 1e-12 && std::abs(d1) < 1e-12)
                continue;
            if (d0 * d1 > 0.0)
                continue; // same side, no intersection on that edge

            // compute intersection point (linear interpolation)
            double t = d1 / (d1 - d0);             // following Rust pm = p0 * (d1/(d1-d0)) + p1 * (d0/(d0-d1))
            Vector2d pm = p0 * t + p1 * (1.0 - t); // watch algebra: derived equivalent
            double t0 = line_t.dot(pm);

            // compute info merging like Rust:
            array<size_t, 4> info0 = cell.vtx2info[i0];
            array<size_t, 4> info1 = cell.vtx2info[i1];

            set<size_t> set_a = {info0[2], info0[3]};
            set<size_t> set_b = {info1[2], info1[3]};
            set<size_t> intersec;
            for (auto v : set_a)
                if (set_b.count(v))
                    intersec.insert(v);
            intersec.erase(std::numeric_limits<size_t>::max());

            array<size_t, 4> info;
            if (intersec.empty()){
                info = {info0[0], i_vtx, j_vtx, std::numeric_limits<size_t>::max()};
            }
            else if (intersec.size() == 1){
                size_t k_vtx = *intersec.begin();
                info = {std::numeric_limits<size_t>::max(), i_vtx, k_vtx, j_vtx};
            }
            else{
                // unexpected
                throw std::runtime_error("cut_polygon_by_line: intersection info ambiguous");
            }
            // Note: Rust used -t0 for sorting; we keep same monotonic ordering
            vtxnews.emplace_back(-t0, i0, pm, info);
        }

        // no intersections
        if (vtxnews.empty()){
            if (is_inside)
                return {cell};
            else
                return {};
        }

        // sort by first element (the -t0) ascending
        std::sort(vtxnews.begin(), vtxnews.end(),
                  [](auto &a, auto &b)
                  { return std::get<0>(a) < std::get<0>(b); });

        assert(vtxnews.size() % 2 == 0);

        // build mapping from original vertex -> index of vtxnews (if any)
        vector<size_t> vtx2vtxnew(cell.vtx2xy.size(), std::numeric_limits<size_t>::max());
        for (size_t i = 0; i < vtxnews.size(); ++i){
            size_t orig_idx = std::get<1>(vtxnews[i]);
            vtx2vtxnew[orig_idx] = i;
        }

        vector<char> vtxnew2isvisited(vtxnews.size(), 0);
        vector<Cell> cells;
        while (true){
            auto opt = hoge(cell.vtx2xy, cell.vtx2info, vtxnews, vtx2vtxnew, vtxnew2isvisited);
            if (!opt.has_value())
                break;
            cells.push_back(std::move(opt.value()));
        }
        return cells;
    }

    // ----------------- voronoi_cells -----------------
    vector<Cell> voronoi_cells(
        const vector<double> &vtxl2xy_flat,
        const vector<double> &site2xy_flat,
        const std::function<bool(size_t)> &site2isalive){
        vector<Vector2d> vtxl2xy = to_vec2_array(vtxl2xy_flat);
        vector<Vector2d> site2xy = to_vec2_array(site2xy_flat);

        size_t num_site = site2xy.size();
        vector<Cell> site2cell(num_site, Cell::new_empty());

        for (size_t i_site = 0; i_site < num_site; ++i_site){
            if (!site2isalive(i_site))
                continue;
            vector<Cell> cell_stack;
            cell_stack.push_back(Cell::new_from_polyloop2(vtxl2xy));
            for (size_t j_site = 0; j_site < num_site; ++j_site){
                if (!site2isalive(j_site))
                    continue;
                if (j_site == i_site)
                    continue;

                Vector2d line_s = 0.5 * (site2xy[i_site] + site2xy[j_site]);
                Vector2d line_n = (site2xy[j_site] - site2xy[i_site]).normalized();

                vector<Cell> cell_stack_new;
                for (auto &cell_in : cell_stack){
                    auto parts = cut_polygon_by_line(cell_in, line_s, line_n, i_site, j_site);
                    cell_stack_new.insert(cell_stack_new.end(), parts.begin(), parts.end());
                }
                cell_stack.swap(cell_stack_new);
                if (cell_stack.empty())
                    break;
            }

            if (cell_stack.empty()){
                site2cell[i_site] = Cell::new_empty();
                continue;
            }
            if (cell_stack.size() == 1){
                site2cell[i_site] = cell_stack[0];
                continue;
            }
            // choose best by membership or area heuristic (like Rust)
            vector<std::pair<double, size_t>> depthcell;
            for (size_t k = 0; k < cell_stack.size(); ++k){
                bool inside = cell_stack[k].is_inside(site2xy[i_site]);
                double score = inside ? 0.0 : 1.0 / std::max(1e-12, cell_stack[k].area());
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

    // ----------------- indexing -----------------
    VoronoiMesh indexing(const vector<Cell> &site2cell){
        size_t num_site = site2cell.size();

        auto sort_info = [](const array<size_t, 4> &info){
            array<size_t, 4> tmp = info;
            array<size_t, 3> t = {tmp[1], tmp[2], tmp[3]};
            std::sort(t.begin(), t.end());
            return array<size_t, 4>{tmp[0], t[0], t[1], t[2]};
        };

        // mapping info -> global voronoi vertex index
        std::map<array<size_t, 4>, size_t> info2vtxv;
        vector<array<size_t, 4>> vtxv2info;

        for (const auto &cell : site2cell){
            for (const auto &info : cell.vtx2info){
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
        vector<Vector2d> vtxv2xy(num_vtxv, Vector2d::Zero());
        vector<size_t> site2idx;
        site2idx.reserve(num_site + 1);
        site2idx.push_back(0);
        vector<size_t> idx2vtxc;
        idx2vtxc.reserve(num_vtxv * 2);

        for (const auto &cell : site2cell){
            for (size_t ind = 0; ind < cell.vtx2info.size(); ++ind){
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

    // ----------------- position_of_voronoi_vertex -----------------
    Vector2d position_of_voronoi_vertex(
        const array<size_t, 4> &info,
        const vector<double> &vtxl2xy_flat,
        const vector<double> &site2xy_flat){
        // helpers to fetch coordinates
        auto get_vtxl = [&](size_t idx) -> Vector2d{
            size_t off = idx * 2;
            return Vector2d(vtxl2xy_flat[off], vtxl2xy_flat[off + 1]);
        };
        auto get_site = [&](size_t idx) -> Vector2d{
            size_t off = idx * 2;
            return Vector2d(site2xy_flat[off], site2xy_flat[off + 1]);
        };

        if (info[1] == std::numeric_limits<size_t>::max()){
            // original loop vertex
            return get_vtxl(info[0]);
        }
        else if (info[3] == std::numeric_limits<size_t>::max()){
            // intersection of loop edge and bisector between two sites
            size_t num_vtxl = vtxl2xy_flat.size() / 2;
            assert(info[0] < num_vtxl);
            size_t i1_loop = info[0];
            size_t i2_loop = (i1_loop + 1) % num_vtxl;
            Vector2d l1 = get_vtxl(i1_loop);
            Vector2d l2 = get_vtxl(i2_loop);
            Vector2d s1 = get_site(info[1]);
            Vector2d s2 = get_site(info[2]);
            Vector2d mid = 0.5 * (s1 + s2);
            Vector2d bisdir = util::rotate90(s2 - s1);
            // intersection of line (l1 + t*(l2-l1)) with (mid + u * bisdir)
            return line_intersection_param(l1, (l2 - l1), mid, bisdir);
        }
        else{
            // three-site circumcenter
            assert(info[0] == std::numeric_limits<size_t>::max());
            Vector2d s0 = get_site(info[1]);
            Vector2d s1 = get_site(info[2]);
            Vector2d s2 = get_site(info[3]);
            return circumcenter(s0, s1, s2);
        }
    }

}
