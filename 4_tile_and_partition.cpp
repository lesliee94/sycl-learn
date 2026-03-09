// ============================================================
// Day 3: Tiling and Partitioning
// ============================================================
//
// Previously we learned:
//   Layout: coordinate -> index
//   Tensor: data pointer + Layout (a view, not a copy)
//
// Now we learn how to DECOMPOSE tensors:
//   - Tiling:      split a big tensor into smaller tiles
//   - Partitioning: distribute a tile across threads
//
// This is the bridge from "data structure" to "parallel execution":
//   Big matrix  -->  block tiles  -->  per-thread fragments
//
// This lesson covers:
//   1. local_tile: splitting a matrix into block-level tiles
//   2. Selecting individual tiles by coordinate
//   3. local_partition: distributing a tile across threads
//   4. Thread layout: controlling which thread gets which data
//   5. Two-level decomposition: tile then partition (GEMM pattern)
//   6. The K-dimension: tiling for the reduction loop
//   7. Summary: the full data decomposition pipeline
//
// Note: This is a CPU-only demo. All "thread" and "block" concepts
//       are simulated with loops — the layout logic is identical
//       to what runs on GPU.
//
// Compile:
//   icpx -fsycl -I <sycl-tla>/include 4_tile_and_partition.cpp -o 4_tile_and_partition
//
// ============================================================

#include <cute/tensor.hpp>
#include <cstdio>
#include <cstdlib>

using namespace cute;

// ============================================================
// Helper: print Tensor metadata
// ============================================================
template <class TensorT>
void print_info(const char* name, TensorT const& t) {
    printf("\n=== %s ===\n", name);
    printf("  Shape  : "); print(t.shape());  printf("\n");
    printf("  Stride : "); print(t.stride()); printf("\n");
    printf("  Size   : %d\n", int(size(t)));
}

// ============================================================
// Helper: print 2D Tensor values
// ============================================================
template <class TensorT>
void print_2d(const char* name, TensorT const& t) {
    printf("\n--- %s ---\n", name);
    for (int m = 0; m < size<0>(t); ++m) {
        for (int n = 0; n < size<1>(t); ++n) {
            printf("%5.0f ", float(t(m, n)));
        }
        printf("\n");
    }
}

// ============================================================
// Helper: print 1D Tensor values
// ============================================================
template <class TensorT>
void print_1d(const char* name, TensorT const& t) {
    printf("  %s: ", name);
    for (int i = 0; i < size(t); ++i) {
        printf("%.0f ", float(t(i)));
    }
    printf("\n");
}


int main() {
    printf("============================================================\n");
    printf("  CuTe Tiling & Partitioning - Day 3\n");
    printf("============================================================\n");

    // ===========================================================
    // Setup: create a 16x12 matrix (col-major)
    // ===========================================================
    const int M = 16, N = 12;
    float* data = (float*)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; ++i) data[i] = float(i);

    auto tensor = make_tensor(data, make_shape(M, N));
    print_info("Original tensor (16x12)", tensor);
    print_2d("Original tensor", tensor);

    // ===========================================================
    // 1. local_tile: split into block-level tiles
    // ===========================================================
    //
    // Scenario: we want each "block" to process a 4x6 sub-matrix.
    //
    // local_tile(tensor, tile_shape, coord)
    //   - tile_shape: size of each tile
    //   - coord: which tile(s) to select (_ = all)
    //
    // Result shape: (BM, BN, num_tiles_M, num_tiles_N)
    //   BM=4, BN=6, num_tiles_M=16/4=4, num_tiles_N=12/6=2
    //

    printf("\n\n");
    printf("============================================================\n");
    printf("  1. local_tile: block-level tiling\n");
    printf("============================================================\n");

    auto tile_shape = make_shape(Int<4>{}, Int<6>{});

    // Get ALL tiles at once
    auto all_tiles = local_tile(tensor, tile_shape, make_coord(_, _));
    print_info("all_tiles", all_tiles);
    // Expected shape: (4, 6, 4, 2)
    //                  ^^^^  ^^^^
    //                  tile   tile indices

    // ===========================================================
    // 2. Selecting individual tiles
    // ===========================================================

    printf("\n\n");
    printf("============================================================\n");
    printf("  2. Selecting individual tiles\n");
    printf("============================================================\n");

    // Method A: index into all_tiles
    auto tile_00 = all_tiles(_, _, 0, 0);  // top-left tile
    auto tile_10 = all_tiles(_, _, 1, 0);  // second row, first col
    auto tile_01 = all_tiles(_, _, 0, 1);  // first row, second col
    auto tile_31 = all_tiles(_, _, 3, 1);  // bottom-right tile

    print_info("tile(0,0) — top-left", tile_00);
    print_2d("tile(0,0)", tile_00);

    print_info("tile(1,0) — second row", tile_10);
    print_2d("tile(1,0)", tile_10);

    print_info("tile(0,1) — top-right half", tile_01);
    print_2d("tile(0,1)", tile_01);

    print_info("tile(3,1) — bottom-right", tile_31);
    print_2d("tile(3,1)", tile_31);

    // Method B: directly select via local_tile
    // Equivalent to all_tiles(_, _, 1, 0)
    auto tile_10_direct = local_tile(tensor, tile_shape, make_coord(1, 0));
    print_info("tile(1,0) via direct coord", tile_10_direct);
    print_2d("tile(1,0) direct", tile_10_direct);

    // ===========================================================
    // 3. local_partition: distributing a tile across threads
    // ===========================================================
    //
    // Now we have a 4x6 tile. We want to split it among threads.
    //
    // Key idea: define a "thread layout" that describes
    //   how threads are arranged over the tile.
    //
    // Thread layout (2, 3) means:
    //   2 threads along M, 3 threads along N
    //   Total: 6 threads
    //
    // Each thread gets a (4/2, 6/3) = (2, 2) sub-tile.
    //

    printf("\n\n");
    printf("============================================================\n");
    printf("  3. local_partition: per-thread data\n");
    printf("============================================================\n");

    // Use tile(0,0) as our working tile
    auto tile = tile_00;
    print_2d("Working tile (0,0)", tile);

    // Define thread layout: 2 threads in M, 3 threads in N
    auto thr_layout = make_layout(make_shape(Int<2>{}, Int<3>{}));
    //
    // Thread assignment (col-major by default):
    //   Thread 0 = (0,0)    Thread 2 = (0,1)    Thread 4 = (0,2)
    //   Thread 1 = (1,0)    Thread 3 = (1,1)    Thread 5 = (1,2)
    //
    // Each thread gets a (4/2, 6/3) = (2, 2) fragment
    //

    printf("\nThread layout: "); print(thr_layout); printf("\n");
    printf("Each thread gets: (%d, %d) elements\n",
           M / int(size<0>(thr_layout)) / (M/int(size<0>(tile_shape))),
           N / int(size<1>(thr_layout)) / (N/int(size<1>(tile_shape))));

    // Show what each thread gets
    for (int tid = 0; tid < size(thr_layout); ++tid) {
        auto fragment = local_partition(tile, thr_layout, tid);
        char name[64];
        snprintf(name, sizeof(name), "Thread %d fragment", tid);
        print_info(name, fragment);
        print_2d(name, fragment);
    }

    // ===========================================================
    // 4. Thread layout matters: coalescing implications
    // ===========================================================
    //
    // The thread layout determines which thread accesses which
    // memory addresses. For GPU coalescing:
    //
    //   Adjacent threads should access adjacent addresses.
    //   = Adjacent threads should differ in the stride-1 dimension.
    //
    // For col-major data (stride-1 along M):
    //   Good: threads spread along M first  -> (num_thr_M, num_thr_N)
    //   Bad:  threads spread along N first  -> (num_thr_N, num_thr_M)
    //

    printf("\n\n");
    printf("============================================================\n");
    printf("  4. Thread layout and coalescing\n");
    printf("============================================================\n");

    // Good layout for col-major: more threads along M
    auto thr_good = make_layout(make_shape(Int<4>{}, Int<1>{}));
    // 4 threads along M (stride=1 dim), 1 along N
    // Thread 0 -> rows 0,4,8,12;  Thread 1 -> rows 1,5,9,13; ...
    // Adjacent threads access adjacent memory -> coalesced!

    printf("\nGood thread layout for col-major: ");
    print(thr_good); printf("\n");
    for (int tid = 0; tid < size(thr_good); ++tid) {
        auto frag = local_partition(tile, thr_good, tid);
        char name[64];
        snprintf(name, sizeof(name), "  thr %d", tid);
        print_info(name, frag);
    }

    // Bad layout for col-major: threads along N
    auto thr_bad = make_layout(make_shape(Int<1>{}, Int<4>{}));
    // 1 thread along M, 4 along N
    // Adjacent threads access addresses M apart -> not coalesced!

    printf("\nBad thread layout for col-major: ");
    print(thr_bad); printf("\n");
    for (int tid = 0; tid < size(thr_bad); ++tid) {
        auto frag = local_partition(tile, thr_bad, tid);
        char name[64];
        snprintf(name, sizeof(name), "  thr %d", tid);
        print_info(name, frag);
    }

    // ===========================================================
    // 5. Two-level decomposition: the GEMM pattern
    // ===========================================================
    //
    // Real GPU kernels use two levels:
    //
    //   Level 1: big matrix -> block tiles     (local_tile)
    //   Level 2: block tile -> thread fragments (local_partition)
    //
    // Full pipeline:
    //   for each block (bm, bn):
    //     tile = local_tile(matrix, tile_shape, (bm, bn))
    //     for each thread tid:
    //       frag = local_partition(tile, thr_layout, tid)
    //       // compute on frag
    //

    printf("\n\n");
    printf("============================================================\n");
    printf("  5. Two-level decomposition (GEMM pattern)\n");
    printf("============================================================\n");

    auto block_shape = make_shape(Int<4>{}, Int<6>{});
    auto thread_layout = make_layout(make_shape(Int<2>{}, Int<3>{}));

    int num_blocks_m = M / size<0>(block_shape);  // 16/4 = 4
    int num_blocks_n = N / size<1>(block_shape);  // 12/6 = 2

    printf("Grid: %d x %d blocks\n", num_blocks_m, num_blocks_n);
    printf("Block tile: (%d, %d)\n", int(size<0>(block_shape)), int(size<1>(block_shape)));
    printf("Threads per block: %d\n", int(size(thread_layout)));

    // Simulate block (1, 0)
    int bm = 1, bn = 0;
    printf("\n--- Simulating block (%d, %d) ---\n", bm, bn);

    auto block_tile = local_tile(tensor, block_shape, make_coord(bm, bn));
    print_2d("Block tile", block_tile);

    // Simulate thread 0 in this block
    int tid = 0;
    auto thr_frag = local_partition(block_tile, thread_layout, tid);
    printf("\nThread %d in block (%d,%d):\n", tid, bm, bn);
    print_info("  fragment", thr_frag);
    print_2d("  fragment values", thr_frag);

    // Verify: thread 0 should get the top-left (2,2) of block (1,0)
    printf("\nVerification:\n");
    printf("  block_tile(0,0) = %.0f\n", block_tile(0,0));
    printf("  thr_frag(0,0)   = %.0f  (should match)\n", thr_frag(0,0));

    // ===========================================================
    // 6. The K-dimension: tiling for reduction
    // ===========================================================
    //
    // In GEMM:  C(M,N) += A(M,K) * B(K,N)
    //
    // The K dimension is special — it's the reduction axis.
    // We tile K into chunks and loop over them:
    //
    //   for k_tile in range(num_k_tiles):
    //     load A_tile(BM, BK) from global
    //     load B_tile(BK, BN) from global
    //     C_tile += A_tile * B_tile
    //
    // With local_tile, the K dim uses _ to get all tiles:
    //   local_tile(A, (BM, BK), (bm, _))
    //   Result: (BM, BK, num_k_tiles)
    //           -> loop over the last dim
    //

    printf("\n\n");
    printf("============================================================\n");
    printf("  6. K-dimension tiling (reduction loop)\n");
    printf("============================================================\n");

    // Simulate A: 16x8 matrix
    const int K = 8;
    float* data_A = (float*)malloc(M * K * sizeof(float));
    for (int i = 0; i < M * K; ++i) data_A[i] = float(i);

    auto A = make_tensor(data_A, make_shape(M, K));
    print_info("A (16x8)", A);

    auto BM = Int<4>{};
    auto BK = Int<4>{};

    // For block row bm=1: get all K-tiles
    // local_tile(A, (BM, BK), (bm, _))
    // Result shape: (4, 4, 2)  — two K-tiles
    //                      ^--- num_k_tiles = 8/4 = 2
    auto a_tiles = local_tile(A, make_shape(BM, BK), make_coord(1, _));
    print_info("A tiles for block row 1", a_tiles);

    printf("\nK-tile loop:\n");
    int num_k_tiles = size<2>(a_tiles);
    for (int k = 0; k < num_k_tiles; ++k) {
        auto a_k = a_tiles(_, _, k);
        char name[64];
        snprintf(name, sizeof(name), "A_tile (bm=1, k=%d)", k);
        print_2d(name, a_k);
    }

    // ===========================================================
    // 7. Summary
    // ===========================================================
    printf("\n");
    printf("============================================================\n");
    printf("  Summary\n");
    printf("============================================================\n");
    printf("\n");
    printf("  Tiling pipeline for GPU GEMM:\n");
    printf("\n");
    printf("    1. local_tile(matrix, (BM,BN), (bm,bn))\n");
    printf("       -> block tile: each thread block gets one\n");
    printf("\n");
    printf("    2. local_partition(tile, thr_layout, tid)\n");
    printf("       -> thread fragment: each thread gets its share\n");
    printf("\n");
    printf("    3. local_tile(A, (BM,BK), (bm,_))\n");
    printf("       -> K-tiled: loop over reduction dimension\n");
    printf("\n");
    printf("  Key rules:\n");
    printf("    - Thread layout must align with data layout for coalescing\n");
    printf("    - Col-major data: spread threads along M (stride-1 dim)\n");
    printf("    - Row-major data: spread threads along N (stride-1 dim)\n");
    printf("    - Tile sizes should be multiples of thread count per dim\n");
    printf("\n");
    printf("  What we did NOT cover yet:\n");
    printf("    - Copy atoms (efficient global->shared memory transfer)\n");
    printf("    - MMA atoms (mapping to tensor core instructions)\n");
    printf("    - Swizzled shared memory layouts (bank conflict avoidance)\n");
    printf("\n");
    printf("  Next: Copy & MMA — hardware-aware tiling\n");
    printf("============================================================\n");

    free(data);
    free(data_A);
    return 0;
}

// ============================================================
//  Concept Reference:
// ============================================================
//
//  Operation                    What it does
//  ---------------------------  -------------------------------------------
//  local_tile(t, shape, coord)  Split tensor into tiles of given shape
//                               coord selects which tile(s): _ = all
//
//  local_partition(t, thr, id)  Split tensor among threads according to
//                               thread layout; id selects this thread's part
//
//  Two-level decomposition:     local_tile (block) + local_partition (thread)
//                               = standard GEMM tiling pattern
//
//  K-tiling:                    local_tile(A, (BM,BK), (bm, _))
//                               last dim = K-tiles, loop over it
//
//  Thread layout choice:        Must match data layout for coalescing
//                               col-major -> threads along M (stride-1)
//                               row-major -> threads along N (stride-1)
//
