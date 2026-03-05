// ============================================================
//  SYCL 向量加法 —— 对照 CUDA 的写法逐一讲解
// ============================================================
//
//  CUDA 写法回顾：
//  -------------------------------------------------------
//  __global__ void add(float* a, float* b, float* c) {
//      int id = blockIdx.x * blockDim.x + threadIdx.x;
//      c[id] = a[id] + b[id];
//  }
//  add<<<grid, block>>>(a, b, c);
//  -------------------------------------------------------
//
//  SYCL 不需要 __global__，kernel 就是一个 lambda。
//  下面用两种方式实现：
//    方式1: parallel_for(range)    —— 简单模式，类似"只有1个block"
//    方式2: parallel_for(nd_range) —— 完整模式，对应 CUDA 的 grid/block
//

#include <sycl/sycl.hpp>
#include <iostream>
using namespace sycl;

int main() {
    queue q(gpu_selector_v);
    std::cout << "Running on: "
              << q.get_device().get_info<info::device::name>() << "\n";

    const int N = 1024;

    // ---- 分配内存 (USM) ----
    // CUDA:  cudaMalloc(&d_a, N*sizeof(float));
    // SYCL:  malloc_device<float>(N, q);
    float* A = malloc_device<float>(N, q);
    float* B = malloc_device<float>(N, q);
    float* C = malloc_device<float>(N, q);

    // 用 host 端临时数组初始化
    float* h_A = malloc_host<float>(N, q);
    float* h_B = malloc_host<float>(N, q);
    float* h_C = malloc_host<float>(N, q);

    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    // 拷贝到 GPU
    q.memcpy(A, h_A, N * sizeof(float));
    q.memcpy(B, h_B, N * sizeof(float));
    q.wait();  // 等两个 memcpy 都完成

    // ============================================================
    // 方式1: parallel_for(range<1>(N), ...)
    // ============================================================
    // 这是最简单的写法，你只告诉运行时"我要 N 个线程"
    // 运行时自动决定怎么分 block/grid
    //
    // CUDA 对比:
    //   __global__ void add(float* a, float* b, float* c) {
    //       int i = blockIdx.x * blockDim.x + threadIdx.x;  // 手动算全局id
    //       c[i] = a[i] + b[i];
    //   }
    //
    // SYCL:
    //   lambda 的参数 id<1> i 直接就是全局线程id，不用自己算
    //
    q.submit([&](handler& h) {
        h.parallel_for(range<1>(N), [=](id<1> i) {
            // id<1> i  ←→  CUDA 的 blockIdx.x * blockDim.x + threadIdx.x
            // 直接就是全局索引，不需要手动计算
            C[i] = A[i] + B[i];
        });
    }).wait();

    // 验证方式1的结果
    q.memcpy(h_C, C, N * sizeof(float)).wait();
    std::cout << "[方式1] C[0]=" << h_C[0] << ", C[1]=" << h_C[1]
              << ", C[1023]=" << h_C[1023] << "\n";

    // ============================================================
    // 方式2: parallel_for(nd_range, ...)  —— 对应 CUDA grid/block
    // ============================================================
    // 当你需要控制 work-group 大小（= CUDA block 大小）、
    // 或者需要用 local memory（= CUDA shared memory）时用这个
    //
    //  CUDA:  add<<<grid, block>>>(a, b, c);
    //         grid  = N / 256;     // block 的数量
    //         block = 256;         // 每个 block 的线程数
    //
    //  SYCL:  nd_range<1>(range<1>(N), range<1>(256))
    //                      ↑ global size (总线程数)   ↑ local size (work-group大小 = CUDA block大小)
    //         注意：SYCL 传的是 global size，不是 grid size!
    //         CUDA 的 grid size = global_size / local_size

    const int BLOCK_SIZE = 256;  // = CUDA 的 blockDim.x

    q.submit([&](handler& h) {
        h.parallel_for(
            nd_range<1>(range<1>(N), range<1>(BLOCK_SIZE)),
            [=](nd_item<1> item) {
                // nd_item 能拿到所有信息，对照 CUDA:
                //
                //   item.get_global_id(0)     ←→  blockIdx.x * blockDim.x + threadIdx.x  (全局id)
                //   item.get_local_id(0)      ←→  threadIdx.x   (block 内的 id)
                //   item.get_group(0)         ←→  blockIdx.x    (第几个 block)
                //   item.get_local_range(0)   ←→  blockDim.x    (block 大小)
                //   item.get_group_range(0)   ←→  gridDim.x     (grid 大小，block 个数)

                int i = item.get_global_id(0);
                C[i] = A[i] + B[i];
            }
        );
    }).wait();

    // 验证方式2的结果
    q.memcpy(h_C, C, N * sizeof(float)).wait();
    std::cout << "[方式2] C[0]=" << h_C[0] << ", C[1]=" << h_C[1]
              << ", C[1023]=" << h_C[1023] << "\n";

    // ---- 释放 ----
    free(A, q); free(B, q); free(C, q);
    free(h_A, q); free(h_B, q); free(h_C, q);

    return 0;
}

// ============================================================
//  总结对照表:
// ============================================================
//
//  CUDA                          SYCL
//  ----------------------------  ----------------------------------
//  __global__ void kernel()      lambda 写在 parallel_for 里
//  kernel<<<grid, block>>>()     q.submit + h.parallel_for(nd_range)
//  threadIdx.x                   item.get_local_id(0)
//  blockIdx.x                    item.get_group(0)
//  blockDim.x                    item.get_local_range(0)
//  gridDim.x                     item.get_group_range(0)
//  全局id(手动算)                 item.get_global_id(0) 或 id<1> i
//  cudaMalloc                    malloc_device
//  cudaMallocHost                malloc_host
//  cudaMemcpy                    q.memcpy
//  __shared__                    local_accessor (local memory)
//  __syncthreads()               item.barrier(access::fence_space::local_space)
//
