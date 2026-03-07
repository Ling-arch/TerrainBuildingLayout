#pragma once

#include <vector>
#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif
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

void computeFlowAccumulationMFD_CUDA(
    const int *order,
    const int *neighborCounts,
    const int *neighborFaces,
    const float *weights,
    int N,
    int maxNeighbors,
    float *flowHost);