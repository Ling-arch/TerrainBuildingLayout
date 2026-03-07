#include "viewshed_cuda.h"
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/fill.h>
#include <thrust/device_vector.h>
#include <algorithm>
#include <stdio.h>
#define BLOCK_SIZE 256

__device__ bool sampleHeight_device(
    float &outHeight,
    float px,
    float py,
    const float *vertex_h,
    int width,
    int height,
    float cellSize)
{
    float fx = (px + width * cellSize * 0.5f) / cellSize;
    float fy = (py + height * cellSize * 0.5f) / cellSize;

    int gx = floorf(fx);
    int gy = floorf(fy);

    if (gx < 0 || gy < 0 || gx >= width || gy >= height)
        return false;
    int gridIdx = width * gy + gx;
    int v2 = gridIdx + gy;
    int v3 = v2 + 1;
    int v0 = v2 + (width + 1);
    int v1 = v0 + 1;

    float h00 = vertex_h[v0];
    float h10 = vertex_h[v1];
    float h01 = vertex_h[v2];
    float h11 = vertex_h[v3];

    float tx = fx - gx;
    float ty = 1 - fy + gy;

    float w00 = (1 - tx) * (1 - ty);
    float w10 = tx * (1 - ty);
    float w01 = (1 - tx) * ty;
    float w11 = tx * ty;

    outHeight = h00 * w00 + h10 * w10 + h01 * w01 + h11 * w11;

    return true;
}

__device__ bool is_occluded(
    float ox, float oy, float oz,
    float tx, float ty, float tz,
    const float *vertex_h,
    int width,
    int height,
    float cellSize)
{
    float dx = tx - ox;
    float dy = ty - oy;

    float dist = sqrtf(dx * dx + dy * dy);

    int steps = max(1, (int)(dist / cellSize));

    float step_x = dx / steps;
    float step_y = dy / steps;

    float slope_target = (tz - oz) / dist;

    float px = ox;
    float py = oy;

    float max_slope = -1e30f;

    for (int i = 1; i < steps; i++)
    {
        px += step_x;
        py += step_y;

        float terrain_h;

        if (!sampleHeight_device(
                terrain_h,
                px,
                py,
                vertex_h,
                width,
                height,
                cellSize))
            continue;

        float d = sqrtf((px - ox) * (px - ox) + (py - oy) * (py - oy));

        float slope = (terrain_h - oz) / d;

        if (slope > max_slope)
            max_slope = slope;

        if (max_slope > slope_target)
            return true;
    }

    return false;
}

// -------------------------------------------
// kernel
// -------------------------------------------


__global__ void viewshed_kernel(
    const float *face_xyz,
    const float *vertex_h,
    int face_count,
    int width,
    int height,
    float cellSize,
    float observerH,
    int *out_view_nums)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= face_count)
        return;

    float ox = face_xyz[tid * 3 + 0];
    float oy = face_xyz[tid * 3 + 1];
    float oz = face_xyz[tid * 3 + 2] + observerH;

    int visible = 0;

    for (int target = 0; target < face_count; target++)
    {
        if (target == tid)
            continue;

        float tx = face_xyz[target * 3 + 0];
        float ty = face_xyz[target * 3 + 1];
        float tz = face_xyz[target * 3 + 2];

        if (!is_occluded(
                ox, oy, oz,
                tx, ty, tz,
                vertex_h,
                width,
                height,
                cellSize))
        {
            visible++;
        }
    }

    out_view_nums[tid] = visible;
}


// -------------------------------------------
// host wrapper
// -------------------------------------------
void compute_viewshed_cuda(
    const float *face_xyz,
    const float *vertex_h,
    int face_count,
    int width,
    int height,
    float cellSize,
    float observerH,
    int *out_view_nums)
{
    float *d_face_xyz;
    float *d_vertex_h;
    int *d_out;

    size_t face_size = face_count * 3 * sizeof(float);
    size_t vertex_size = (width + 1) * (height + 1) * sizeof(float);
    size_t out_size = face_count * sizeof(int);

    cudaMalloc(&d_face_xyz, face_size);
    cudaMalloc(&d_vertex_h, vertex_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_face_xyz, face_xyz, face_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vertex_h, vertex_h, vertex_size, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (face_count + block - 1) / block;

    viewshed_kernel<<<grid, block>>>(
        d_face_xyz,
        d_vertex_h,
        face_count,
        width,
        height,
        cellSize,
        observerH,
        d_out);

    cudaMemcpy(out_view_nums, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_face_xyz);
    cudaFree(d_vertex_h);
    cudaFree(d_out);
}

// kernel：单观察点计算所有face可见性
__global__ void point_viewshed_kernel(
    const float *face_xyz, // Nx3
    int N,
    const float *terrain_h,
    int grid_w,
    int grid_h,
    float cellSize,
    float obs_x,
    float obs_y,
    float obs_z,
    int *out_visible) // 输出 0/1
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    float tx = face_xyz[tid * 3 + 0];
    float ty = face_xyz[tid * 3 + 1];
    float tz = face_xyz[tid * 3 + 2];

    if (!is_occluded(obs_x, obs_y, obs_z, tx, ty, tz, terrain_h, grid_w, grid_h, cellSize))
        out_visible[tid] = 1;
    else
        out_visible[tid] = 0;
}

void point_viewshed_cuda(
    const float *face_xyz, // host Nx3
    int N,
    const float *terrain_h, // host grid_w * grid_h
    int grid_w,
    int grid_h,
    float cellSize,
    float obs_x,
    float obs_y,
    float obs_z,
    int *out_visible) // host N
{
    float *d_face_xyz = nullptr;
    float *d_terrain_h = nullptr;
    int *d_visible = nullptr;

    size_t face_bytes = N * 3 * sizeof(float);
    size_t terrain_bytes = grid_w * grid_h * sizeof(float);
    size_t vis_bytes = N * sizeof(int);

    // -------- GPU 内存申请 --------
    cudaMalloc(&d_face_xyz, face_bytes);
    cudaMalloc(&d_terrain_h, terrain_bytes);
    cudaMalloc(&d_visible, vis_bytes);

    // -------- 拷贝数据到 GPU --------
    cudaMemcpy(d_face_xyz, face_xyz, face_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_terrain_h, terrain_h, terrain_bytes, cudaMemcpyHostToDevice);

    // -------- kernel 启动参数 --------
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // -------- kernel launch --------
    point_viewshed_kernel<<<blocks, threads>>>(
        d_face_xyz,
        N,
        d_terrain_h,
        grid_w,
        grid_h,
        cellSize,
        obs_x,
        obs_y,
        obs_z,
        d_visible);

    // 等待 GPU 完成
    cudaDeviceSynchronize();

    // -------- 拷贝结果回 CPU --------
    cudaMemcpy(out_visible, d_visible, vis_bytes, cudaMemcpyDeviceToHost);

    // -------- 释放 GPU 内存 --------
    cudaFree(d_face_xyz);
    cudaFree(d_terrain_h);
    cudaFree(d_visible);
}

__global__ void flowMFDKernel(
    const float *faceZ,
    const int *neighborCounts,
    const int *neighborFaces,
    const float *weights,
    int N,
    int maxNeighbors,
    float *flowAccumulation)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    // 初始化每个面水量为1
    flowAccumulation[tid] = 1.0f;
    __syncthreads();
}

// ----------------------------
// CUDA 汇水量迭代更新
// ----------------------------
__global__ void flowMFDIterKernel(
    const int *neighborCounts,
    const int *neighborFaces,
    const float *weights,
    int N,
    int maxNeighbors,
    float *flowAccumulation,
    float *flowTmp)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N)
        return;

    float acc = flowAccumulation[tid]; // 本面水量
    int nc = neighborCounts[tid];

    for (int k = 0; k < nc; ++k)
    {
        int idx = tid * maxNeighbors + k;
        int n = neighborFaces[idx];
        float w = weights[idx];
        atomicAdd(&flowTmp[n], acc * w);
    }
}

__global__ void flowPropagationKernel(
    const int *order,
    const int *neighborCounts,
    const int *neighborFaces,
    const float *weights,
    int N,
    int maxNeighbors,
    float *flow)
{
    int step = blockIdx.x * blockDim.x + threadIdx.x;

    if (step >= N)
        return;

    int i = order[step];

    float water = flow[i];

    int count = neighborCounts[i];

    for (int k = 0; k < count; ++k)
    {
        int idx = i * maxNeighbors + k;

        int n = neighborFaces[idx];

        if (n < 0)
            continue;

        float w = weights[idx];

        atomicAdd(&flow[n], water * w);
    }
}
// ----------------------------
// 外部接口
// ----------------------------
void computeFlowAccumulationMFD_CUDA(
    const int *order,
    const int *neighborCounts,
    const int *neighborFaces,
    const float *weights,
    int N,
    int maxNeighbors,
    float *flowHost)
{
    int *d_order;
    int *d_neighborCounts;
    int *d_neighborFaces;
    float *d_weights;
    float *d_flow;

    cudaMalloc(&d_order, sizeof(int) * N);
    cudaMalloc(&d_neighborCounts, sizeof(int) * N);
    cudaMalloc(&d_neighborFaces, sizeof(int) * N * maxNeighbors);
    cudaMalloc(&d_weights, sizeof(float) * N * maxNeighbors);
    cudaMalloc(&d_flow, sizeof(float) * N);

    cudaMemcpy(d_order, order, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighborCounts, neighborCounts, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighborFaces, neighborFaces, sizeof(int) * N * maxNeighbors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, sizeof(float) * N * maxNeighbors, cudaMemcpyHostToDevice);

    cudaMemset(d_flow, 0, sizeof(float) * N);

    // 每个face初始降雨=1
    std::vector<float> init(N, 1.0f);
    cudaMemcpy(d_flow, init.data(), sizeof(float) * N, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    flowPropagationKernel<<<blocks, threads>>>(
        d_order,
        d_neighborCounts,
        d_neighborFaces,
        d_weights,
        N,
        maxNeighbors,
        d_flow);

    cudaMemcpy(flowHost, d_flow, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(d_order);
    cudaFree(d_neighborCounts);
    cudaFree(d_neighborFaces);
    cudaFree(d_weights);
    cudaFree(d_flow);
}
