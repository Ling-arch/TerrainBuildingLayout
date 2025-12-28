#include "loss.h"

namespace loss
{

    // num_group,  site2group,   room2group
    std::tuple<size_t, std::vector<size_t>, std::vector<std::vector<size_t>>> topology(
        const VoronoiInfo &voronoi_info,
        size_t num_room,
        const std::vector<size_t> &site2room)
    {
        // ---------- 1. 定义“是否可连通”的回调 ----------
        auto siteface2adjsitesameroom = [&](size_t i_site, size_t i_face) -> size_t
        {
            size_t i_room = site2room[i_site];
            if (i_room == INVALID)
                return INVALID;

            size_t j_site = voronoi_info.idx2site[voronoi_info.site2idx[i_site] + i_face];
            if (j_site == INVALID)
                return INVALID;

            size_t j_room = site2room[j_site];
            assert(j_room != INVALID);

            if (i_room != j_room)
                return INVALID;

            return j_site;
        };

        // ---------- 2. 对 site 做连通分组 ----------
        size_t num_group;
        std::vector<size_t> site2group;

        std::tie(num_group, site2group) = elem2group_from_polygon_mesh(voronoi_info.site2idx, siteface2adjsitesameroom);

        assert(site2group.size() == site2room.size());

        // ---------- 3. room -> group ----------
        std::vector<std::set<size_t>> room2group_set(num_room);

        for (size_t i_site = 0; i_site < site2room.size(); ++i_site)
        {
            size_t i_room = site2room[i_site];
            if (i_room == INVALID)
                continue;

            size_t i_group = site2group[i_site];
            room2group_set[i_room].insert(i_group);
        }

        // ---------- 4. set -> vector ----------
        std::vector<std::vector<size_t>> room2group(num_room);
        for (size_t i = 0; i < num_room; ++i)
        {
            room2group[i].assign(
                room2group_set[i].begin(),
                room2group_set[i].end());
        }

        return {num_group, site2group, room2group};
    }

    std::vector<std::vector<size_t>> inverse_map(size_t num_group, const std::vector<size_t> &site2group)
    {
    
        // group -> ordered unique sites
        std::vector<std::set<size_t>> group2site_set(num_group);

        for (size_t i_site = 0; i_site < site2group.size(); ++i_site)
        {
            size_t i_group = site2group[i_site];
            if (i_group == INVALID)
                continue;

            group2site_set[i_group].insert(i_site);
        }

        // convert set -> vector
        std::vector<std::vector<size_t>> group2site(num_group);
        for (size_t i = 0; i < num_group; ++i)
        {
            group2site[i].assign(
                group2site_set[i].begin(),
                group2site_set[i].end());
        }

        return group2site;
    }

    bool is_two_room_connected(
        size_t i0_room,
        size_t i1_room,
        const std::vector<size_t> &site2room,
        const std::vector<std::vector<size_t>> &room2site,
        const VoronoiInfo &voronoi_info)
    {
        for (size_t i_site : room2site[i0_room])
        {
            size_t beg = voronoi_info.site2idx[i_site];
            size_t end = voronoi_info.site2idx[i_site + 1];

            for (size_t k = beg; k < end; ++k)
            {
                size_t j_site = voronoi_info.idx2site[k];
                if (j_site == INVALID)
                    continue;

                if (site2room[j_site] == i1_room)
                    return true;
            }
        }
        return false;
    }

    std::pair<size_t, size_t> find_nearest_site(
        size_t i0_room,
        size_t i1_room,
        const std::vector<std::vector<size_t>> &room2site,
        const std::vector<float> &site2xy) // flattened [x0,y0,x1,y1,...]
    {

        std::pair<size_t, size_t> pair{0, 0};
        float min_dist = std::numeric_limits<float>::infinity();

        for (size_t i_site : room2site[i0_room])
        {
            Vector2 pi(site2xy[2 * i_site], site2xy[2 * i_site + 1]);

            for (size_t j_site : room2site[i1_room])
            {
                Vector2 pj(site2xy[2 * j_site], site2xy[2 * j_site + 1]);
                float dist = (pi - pj).norm();

                if (dist < min_dist)
                {
                    min_dist = dist;
                    pair = {i_site, j_site};
                }
            }
        }
        return pair;
    }

    torch::Tensor unidirectional(
        const torch::Tensor &site2xy,
        const std::vector<size_t> &site2room,
        size_t num_room,
        const VoronoiInfo &voronoi_info,
        const std::vector<std::pair<size_t, size_t>> &room_connections)
    {
        TORCH_CHECK(site2xy.dim() == 2 && site2xy.size(1) == 2);
        TORCH_CHECK(site2xy.dtype() == torch::kFloat32);
        TORCH_CHECK(site2xy.device().is_cpu());

        const int64_t num_site = site2xy.size(0);

        // topology
        auto [num_group, site2group, room2group] = topology(voronoi_info, num_room, site2room);

        auto room2site = inverse_map(num_room, site2room);
        auto group2site = inverse_map(num_group, site2group);

        // flatten site2xy
        auto site2xy_flat = site2xy.contiguous().view({-1});
        std::vector<float> site2xy0(
            site2xy_flat.data_ptr<float>(),
            site2xy_flat.data_ptr<float>() + site2xy_flat.numel());

        TORCH_CHECK(site2xy0.size() == num_site * 2);

        std::vector<float> site2xytrg = site2xy0;

        for (size_t i_room = 0; i_room < num_room; ++i_room)
        {
            TORCH_CHECK(!room2group[i_room].empty());

            // ---------------- room is one piece ----------------
            if (room2group[i_room].size() == 1)
            {
                std::vector<size_t> rooms_to_connect;
                for (auto &p : room_connections)
                {
                    if (p.first == i_room)
                        rooms_to_connect.push_back(p.second);
                    else if (p.second == i_room)
                        rooms_to_connect.push_back(p.first);
                }

                for (size_t j_room : rooms_to_connect)
                {
                    bool connected = is_two_room_connected(
                        i_room, j_room, site2room, room2site, voronoi_info);

                    if (connected)
                        continue;

                    auto [i_site, j_site] = find_nearest_site(i_room, j_room, room2site, site2xy0);

                    site2xytrg[i_site * 2 + 0] = site2xy0[j_site * 2 + 0];
                    site2xytrg[i_site * 2 + 1] = site2xy0[j_site * 2 + 1];
                }
            }
            // ---------------- room is split ----------------
            else
            {
                size_t i_group = INVALID;

                for (size_t j_group : room2group[i_room])
                {
                    for (size_t j_site : group2site[j_group])
                    {
                        if (voronoi_info.site2idx[j_site + 1] >
                            voronoi_info.site2idx[j_site])
                        {
                            i_group = j_group;
                            break;
                        }
                    }
                    if (i_group != INVALID)
                        break;
                }

                // no cell
                if (i_group == INVALID)
                {
                    for (size_t j_group : room2group[i_room])
                    {
                        for (size_t j_site : group2site[j_group])
                        {
                            site2xytrg[j_site * 2 + 0] = 0.5f;
                            site2xytrg[j_site * 2 + 1] = 0.5f;
                        }
                    }
                    continue;
                }

                for (size_t j_group : room2group[i_room])
                {
                    if (j_group == i_group)
                        continue;

                    for (size_t j_site : group2site[j_group])
                    {
                        Eigen::Vector2f pj(
                            site2xy0[j_site * 2 + 0],
                            site2xy0[j_site * 2 + 1]);

                        float dist_min = std::numeric_limits<float>::infinity();
                        Eigen::Vector2f pi_min(0.f, 0.f);

                        for (size_t i_site : group2site[i_group])
                        {
                            Eigen::Vector2f pi(
                                site2xy0[i_site * 2 + 0],
                                site2xy0[i_site * 2 + 1]);

                            float dist = (pj - pi).norm();
                            if (dist < dist_min)
                            {
                                dist_min = dist;
                                pi_min = pi;
                            }
                        }

                        site2xytrg[j_site * 2 + 0] = pi_min.x();
                        site2xytrg[j_site * 2 + 1] = pi_min.y();
                    }
                }
            }
        }

        // build tensor
        auto site2xytrg_tensor =
            torch::from_blob(
                site2xytrg.data(),
                {static_cast<int64_t>(num_site), 2},
                torch::TensorOptions().dtype(torch::kFloat32))
                .clone();

        // (site2xy - site2xytrg).sqr().sum()
        return (site2xy - site2xytrg_tensor).pow(2).sum();
    }

    std::vector<size_t> edge2vtvx_wall(const VoronoiInfo &voronoi_info,const std::vector<size_t> &site2room)
    {
        const auto &site2idx = voronoi_info.site2idx;
        const auto &idx2vtxv = voronoi_info.idx2vtxv;

        std::vector<size_t> edge2vtxv;
        // get wall between rooms
        for (size_t i_site = 0; i_site + 1 < site2idx.size(); ++i_site)
        {
            size_t i_room = site2room[i_site];
            if (i_room == INVALID)
            {
                continue;
            }
            size_t beg = site2idx[i_site];
            size_t end = site2idx[i_site + 1];

            size_t num_vtx_in_site = end - beg;

            for (size_t i0_vtx = 0; i0_vtx < num_vtx_in_site; ++i0_vtx)
            {
                size_t idx = beg + i0_vtx;
                size_t i1_vtx = (i0_vtx + 1) % num_vtx_in_site;
                size_t i0_vtxv = idx2vtxv[idx];
                size_t i1_vtxv = idx2vtxv[site2idx[i_site] + i1_vtx];

                size_t j_site = voronoi_info.idx2site[idx];
                if (j_site == INVALID)
                {
                    continue;
                }

                if (i_site >= j_site)
                {
                    continue;
                }

                size_t j_room = site2room[j_site];
                if (i_room == j_room)
                {
                    continue;
                }

                edge2vtxv.push_back(i0_vtxv);
                edge2vtxv.push_back(i1_vtxv);
            }
        }

        return edge2vtxv;
    }

    torch::Tensor loss_lloyd_internal(
        const VoronoiInfo &voronoi_info,
        const std::vector<size_t> &site2room,
        const torch::Tensor &site2xy,
        const torch::Tensor &vtxv2xy)
    {
        TORCH_CHECK(site2xy.device().is_cpu());
        TORCH_CHECK(vtxv2xy.device().is_cpu());
        TORCH_CHECK(site2xy.dim() == 2 && site2xy.size(1) == 2);

        const size_t num_site = site2room.size();
        TORCH_CHECK(voronoi_info.site2idx.size() == num_site + 1);

        const auto &site2idx = voronoi_info.site2idx;

        /* ------------------------------------------------------------
         * 1. site2canmove
         * ------------------------------------------------------------ */
        std::vector<bool> site2canmove(num_site, false);

        for (size_t i_site = 0; i_site + 1 < site2idx.size(); ++i_site)
        {
            // no cell
            if (site2idx[i_site + 1] == site2idx[i_site])
                continue;

            size_t i_room = site2room[i_site];
            if (i_room == INVALID)
                continue;

            size_t num_vtx_in_site = site2idx[i_site + 1] - site2idx[i_site];
            for (size_t i0_vtx = 0; i0_vtx < num_vtx_in_site; ++i0_vtx)
            {
                size_t idx = site2idx[i_site] + i0_vtx;
                size_t j_site = voronoi_info.idx2site[idx];

                if (j_site == INVALID)
                    continue;
                if (i_site >= j_site)
                    continue;

                size_t j_room = site2room[j_site];
                if (i_room == j_room)
                    continue;

                site2canmove[i_site] = true;
            }
        }

        /* ------------------------------------------------------------
         * 2. build mask tensor (num_site,2)
         *    Rust:
         *      if canmove { [0,0] } else { [1,1] }
         * ------------------------------------------------------------ */
        std::vector<float> mask_vec;
        mask_vec.reserve(num_site * 2);

        for (bool canmove : site2canmove)
        {
            if (canmove)
            {
                mask_vec.push_back(0.f);
                mask_vec.push_back(0.f);
            }
            else
            {
                mask_vec.push_back(1.f);
                mask_vec.push_back(1.f);
            }
        }

        auto mask = torch::from_blob(
                        mask_vec.data(),
                        {static_cast<int64_t>(num_site), 2},
                        torch::TensorOptions().dtype(torch::kFloat32))
                        .clone(); // clone to own memory

        /* ------------------------------------------------------------
         * 3. polygonmesh2_to_cogs
         * ------------------------------------------------------------ */
        PolygonMesh2ToCogsLayer polygonmesh2_to_cogs(voronoi_info.site2idx,voronoi_info.idx2vtxv);
        torch::Tensor site2cogs = polygonmesh2_to_cogs.forward(vtxv2xy);

        /* ------------------------------------------------------------
         * 4. diff + mask + loss
         * ------------------------------------------------------------ */
        torch::Tensor diff = site2xy - site2cogs;
        torch::Tensor diffmasked = mask * diff;

        return diffmasked.pow(2).sum();
    }

    torch::Tensor room2area(
        const std::vector<size_t> &site2room,
        size_t num_room,
        const std::vector<size_t> &site2idx,
        const std::vector<size_t> &idx2vtxv,
        const torch::Tensor &vtxv2xy)
    {
        TORCH_CHECK(vtxv2xy.device().is_cpu());
        TORCH_CHECK(vtxv2xy.dim() == 2 && vtxv2xy.size(1) == 2);

        //std::cout << "TORCH_CHECK OK" << std::endl;
        /* ------------------------------------------------------------
         * 1. site2areas via PolygonMesh2ToAreas
         * ------------------------------------------------------------ */
        PolygonMesh2ToAreaLayer polygonmesh2_to_areas(site2idx,idx2vtxv);

        torch::Tensor site2areas = polygonmesh2_to_areas.forward(vtxv2xy);
        //std::cout << "polygonmesh2_to_areas forward OK" << std::endl;
        // reshape to (num_site, 1) for matmul
        site2areas = site2areas.view({site2areas.size(0), 1});
        //std::cout << "site2areas has calculated" << std::endl;
        /* ------------------------------------------------------------
         * 2. build (num_room, num_site) accumulation matrix
         * ------------------------------------------------------------ */
        const size_t num_site = site2room.size();

        std::vector<float> sum_sites_for_rooms(num_room * num_site, 0.0f);

        for (size_t i_site = 0; i_site < num_site; ++i_site)
        {
            size_t i_room = site2room[i_site];
            if (i_room == INVALID)
                continue;

            TORCH_CHECK(i_room < num_room);
            sum_sites_for_rooms[i_room * num_site + i_site] = 1.0f;
        }

        auto room_site_mat = torch::from_blob(
                                 sum_sites_for_rooms.data(),
                                 {(long)num_room, (long)num_site},
                                 torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
                                 .clone(); // owning tensor

        //std::cout << "room_site_mat has calculated" << std::endl;
        /* ------------------------------------------------------------
         * 3. matmul -> room2area
         * ------------------------------------------------------------ */
        return torch::matmul(room_site_mat, site2areas);
    }

    void remove_site_too_close(
        std::vector<size_t> &site2room,
        const torch::Tensor &site2xy // (num_site, 2)
    )
    {
        TORCH_CHECK(site2xy.device().is_cpu());
        TORCH_CHECK(site2xy.dim() == 2);
        TORCH_CHECK(site2xy.size(1) == 2);
        TORCH_CHECK(site2room.size() == (size_t)site2xy.size(0));

        const size_t num_site = site2room.size();

        // flatten -> Vec<f32>
        torch::Tensor flat = site2xy.contiguous().view({-1});
        std::vector<float> site2xy_flat(
            flat.data_ptr<float>(),
            flat.data_ptr<float>() + flat.numel());

        auto get_xy = [&](size_t i)
        {
            size_t idx = 2 * i;
            return torch::tensor(
                {site2xy_flat[idx], site2xy_flat[idx + 1]},
                torch::TensorOptions().dtype(torch::kFloat32));
        };

        for (size_t i_site = 0; i_site < num_site; ++i_site)
        {
            size_t i_room = site2room[i_site];
            if (i_room == INVALID)
                continue;

            auto p_i = get_xy(i_site);

            for (size_t j_site = i_site + 1; j_site < num_site; ++j_site)
            {
                size_t j_room = site2room[j_site];
                if (j_room == INVALID)
                    continue;
                if (i_room != j_room)
                    continue;

                auto p_j = get_xy(j_site);

                float dist = torch::norm(p_i - p_j).item<float>();
                if (dist < 0.02f)
                {
                    site2room[j_site] = INVALID;
                }
            }
        }
    }

    std::vector<size_t> site2room(
        size_t num_site,
        const std::vector<float> &room2area)
    {
        const size_t num_room = room2area.size();

        std::vector<size_t> site2room(num_site, INVALID);

        const size_t num_site_assign = num_site - num_room;

        float area = 0.0f;
        for (float a : room2area)
            area += a;

        // cumulative sum
        std::vector<float> cumsum(num_room);
        {
            float acc = 0.0f;
            for (size_t i = 0; i < num_room; ++i)
            {
                acc += room2area[i];
                cumsum[i] = acc;
            }
        }

        float area_par_site = area / static_cast<float>(num_site_assign);

        size_t i_site_cur = 0;
        float area_cur = 0.0f;

        for (size_t i_room = 0; i_room < num_room; ++i_room)
        {
            // guarantee at least one site per room
            site2room[i_site_cur] = i_room;
            ++i_site_cur;

            while (true)
            {
                area_cur += area_par_site;

                if (i_site_cur >= num_site)
                    break;

                site2room[i_site_cur] = i_room;
                ++i_site_cur;

                if (area_cur > cumsum[i_room])
                    break;
            }

            if (i_site_cur >= num_site)
                break;
        }

        return site2room;
    }

    std::vector<size_t> site2room(
        size_t num_site,
        const std::vector<float> &room2area,
        const std::vector<size_t> &fixed_site_indices,
        const std::vector<size_t> &fixed_rooms)
    {
        const size_t num_room = room2area.size();
        std::vector<size_t> site2room(num_site, INVALID);

        assert(fixed_site_indices.size() == fixed_rooms.size());

        // ---------- 1) apply fixed assignments ----------
        std::vector<bool> site_fixed(num_site, false);
        std::vector<size_t> room_fixed_count(num_room, 0);

        for (size_t i = 0; i < fixed_site_indices.size(); ++i)
        {
            size_t s = fixed_site_indices[i];
            size_t r = fixed_rooms[i];
            assert(s < num_site);
            assert(r < num_room);

            site2room[s] = r;
            site_fixed[s] = true;
            room_fixed_count[r]++;
        }

        // ---------- 2) collect free sites ----------
        std::vector<size_t> free_sites;
        for (size_t i = 0; i < num_site; ++i)
        {
            if (!site_fixed[i])
                free_sites.push_back(i);
        }

        if (free_sites.empty())
            return site2room;

        // ---------- 3) compute remaining area per room ----------
        float total_area = 0.f;
        for (float a : room2area)
            total_area += a;


        const size_t num_site_assign = num_site - num_room;
        float area_per_site = total_area / static_cast<float>(num_site_assign);
        
        std::vector<float> cumsum(num_room);
        {
            float acc = 0.f;
            for (size_t i = 0; i < num_room; ++i)
            {
                acc += room2area[i];
                cumsum[i] = acc;
            }
        }

        // ---------- 4) assign free sites ----------
        size_t free_idx = 0;
        float area_cur = 0.f;

        for (size_t i_room = 0; i_room < num_room; ++i_room)
        {
            // 如果该 room 没有固定 site，优先保证一个
            if (room_fixed_count[i_room] == 0 && free_idx < free_sites.size())
            {
                site2room[free_sites[free_idx]] = i_room;
                free_idx ++;
            }

            while (free_idx < free_sites.size())
            {
                area_cur += area_per_site;
                site2room[free_sites[free_idx]] = i_room;
                free_idx++;

                if (area_cur > cumsum[i_room])
                    break;
            }

            if (free_idx >= free_sites.size())
                break;
        }

        return site2room;
    }

    torch::Tensor loss_lloyd(
        const std::vector<size_t> &elem2idx,
        const std::vector<size_t> &idx2vtx,
        const torch::Tensor &site2xy,
        const torch::Tensor &vtxv2xy)
    {
        // sanity check
        TORCH_CHECK(site2xy.dim() == 2 && site2xy.size(1) == 2,
                    "site2xy must be (N,2)");
        TORCH_CHECK(vtxv2xy.dim() == 2 && vtxv2xy.size(1) == 2,
                    "vtxv2xy must be (M,2)");

        // PolygonMesh -> COGs
        PolygonMesh2ToCogsLayer polygonmesh2_to_cogs(elem2idx, idx2vtx);

        // (num_site, 2)
        torch::Tensor site2cogs = polygonmesh2_to_cogs.forward(vtxv2xy);

        // Lloyd loss: sum_i ||site_i - cog_i||^2
        torch::Tensor diff = site2xy - site2cogs;

        return diff.pow(2).sum();
    }
}