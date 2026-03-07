#pragma once

#include <vector>

void compute_viewshed_cuda(
    const float *face_xyz,
    const float *vertex_h,
    int face_count,
    int width,
    int height,
    float cellSize,
    float observerH,
    int *out_view_nums);

void point_viewshed_cuda(
    const float *face_xyz, // Nx3
    int N,
    const float *terrain_h,
    int grid_w,
    int grid_h,
    float cellSize,
    float obs_x,
    float obs_y,
    float obs_z,
    int *out_visible);