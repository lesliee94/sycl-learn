// ============================================================
// Day 2: CuTe Tensor Basics
// ============================================================
//
// Previously we learned Layout: coordinate -> index
// Now we learn Tensor: data pointer + Layout
//
// Tensor = Engine (where is data) + Layout (how to index)
//
// Think of a Tensor as:
//   - A pointer (base address of the data)
//   - Plus a Layout (mapping from logical coordinates to offsets)
//
// Key insight:
//   tensor(m, n) is essentially:
//     *(base_ptr + layout(m, n))
//   which expands to:
//     *(base_ptr + m * stride_m + n * stride_n)
//
// This lesson covers:
//   1. Creating Tensors from raw pointers
//   2. Reading and writing Tensor elements
//   3. Querying Tensor properties (shape, stride, size, rank)
//   4. Slicing — extracting a row or column
//   5. local_tile — partitioning a Tensor into tiles
//   6. Summary and concept mapping
//
// Compile:
//   icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -I ../sycl-tla/include 3_tensor_basics.cpp -o 3_tensor_basics
//
// ============================================================

#include <cute/tensor.hpp>
#include <cstdio>
#include <cstdlib>

using namespace cute;

// ============================================================
// Helper: print Tensor metadata
// ============================================================
template <class Tensor>
void print_tensor_info(const char* name, Tensor const& t) {
    printf("\n=== %s ===\n", name);
    printf("  Layout : "); print(t.layout()); printf("\n");
    printf("  Shape  : "); print(t.shape());  printf("\n");
    printf("  Stride : "); print(t.stride()); printf("\n");
    printf("  Size   : %d (total elements)\n", int(size(t)));
    printf("  Rank   : %d\n", int(rank(t)));
}

// ============================================================
// Helper: print 2D Tensor values
// ============================================================
template <class Tensor>
void print_tensor_2d(const char* name, Tensor const& t) {
    printf("\n--- %s values ---\n", name);
    for (int m = 0; m < size<0>(t); ++m) {
        for (int n = 0; n < size<1>(t); ++n) {
            printf("%6.1f ", float(t(m, n)));
        }
        printf("\n");
    }
}

// ============================================================
// Helper: print 1D Tensor values
// ============================================================
template <class Tensor>
void print_tensor_1d(const char* name, Tensor const& t) {
    printf("\n--- %s values ---\n", name);
    for (int i = 0; i < size(t); ++i) {
        printf("%6.1f ", float(t(i)));
    }
    printf("\n");
}


int main() {
    printf("============================================================\n");
    printf("  CuTe Tensor Basics - Day 2\n");
    printf("============================================================\n");

    // ----------------------------------------------------------
    // 1. Creating Tensors from raw pointers
    // ----------------------------------------------------------
    //
    // Core concept:
    //   A Tensor does NOT own data! It is merely a "view".
    //   You are responsible for the lifetime of the underlying memory.
    //
    // CUDA analogy:
    //   float* d_A;  cudaMalloc(&d_A, M*N*sizeof(float));
    //   // d_A is the raw pointer; Tensor simply attaches a layout to it.
    //

    const int M = 4, N = 3;
    float* raw_data = (float*)malloc(M * N * sizeof(float));

    // Initialize: 0, 1, 2, ..., 11
    for (int i = 0; i < M * N; ++i) {
        raw_data[i] = float(i);
    }

    printf("\n--- Raw memory layout ---\n");
    printf("raw_data[] = ");
    for (int i = 0; i < M * N; ++i) {
        printf("%.0f ", raw_data[i]);
    }
    printf("\n");

    // ----- Method A: col-major tensor (CuTe default) -----
    //
    // make_tensor(pointer, layout)
    // Default is LayoutLeft = col-major
    //
    // shape = (4, 3), stride = (1, 4)
    //   tensor(m, n) = *(raw_data + m*1 + n*4)
    //
    auto tensor_col = make_tensor(raw_data, make_shape(M, N));

    print_tensor_info("tensor_col (col-major, default)", tensor_col);
    print_tensor_2d("tensor_col", tensor_col);

    // Explanation:
    //   raw_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    //   Col-major treats the first 4 elements as column 0:
    //     col 0: raw_data[0,1,2,3] = 0,1,2,3
    //     col 1: raw_data[4,5,6,7] = 4,5,6,7
    //     col 2: raw_data[8,9,10,11] = 8,9,10,11
    //
    //   Expected output:
    //     0.0   4.0   8.0
    //     1.0   5.0   9.0
    //     2.0   6.0  10.0
    //     3.0   7.0  11.0

    // ----- Method B: row-major tensor -----
    //
    // Explicitly specify LayoutRight
    //
    // shape = (4, 3), stride = (3, 1)
    //   tensor(m, n) = *(raw_data + m*3 + n*1)
    //
    auto tensor_row = make_tensor(raw_data,
                                  make_layout(make_shape(M, N), LayoutRight{}));

    print_tensor_info("tensor_row (row-major)", tensor_row);
    print_tensor_2d("tensor_row", tensor_row);

    // Explanation:
    //   Row-major treats the first 3 elements as row 0:
    //     row 0: raw_data[0,1,2] = 0,1,2
    //     row 1: raw_data[3,4,5] = 3,4,5
    //     row 2: raw_data[6,7,8] = 6,7,8
    //     row 3: raw_data[9,10,11] = 9,10,11
    //
    //   Expected output:
    //     0.0   1.0   2.0
    //     3.0   4.0   5.0
    //     6.0   7.0   8.0
    //     9.0  10.0  11.0

    // ----------------------------------------------------------
    // 2. Reading and writing Tensor elements
    // ----------------------------------------------------------
    //
    // tensor(m, n) returns a reference — read or write.
    // Under the hood: *(base_ptr + layout(m, n))
    //

    printf("\n\n========== 2. Read / Write Tensor ==========\n");

    printf("tensor_col(0, 0) = %.1f\n", tensor_col(0, 0));  // expected: 0
    printf("tensor_col(2, 1) = %.1f\n", tensor_col(2, 1));  // col-major: 2*1 + 1*4 = 6
    printf("tensor_row(2, 1) = %.1f\n", tensor_row(2, 1));  // row-major: 2*3 + 1*1 = 7

    // Write through tensor_col
    tensor_col(0, 0) = 99.0f;
    printf("\nAfter writing tensor_col(0,0) = 99:\n");
    printf("  tensor_col(0,0) = %.1f\n", tensor_col(0, 0));
    printf("  raw_data[0]     = %.1f  (underlying data changed — tensor is a view!)\n",
           raw_data[0]);
    printf("  tensor_row(0,0) = %.1f  (row-major tensor sees the same memory)\n",
           tensor_row(0, 0));

    // Restore
    tensor_col(0, 0) = 0.0f;

    // ----------------------------------------------------------
    // Key takeaway:
    //   tensor_col and tensor_row share the SAME raw_data!
    //   They just interpret the memory with different layouts.
    //   Modifying one affects the other.
    //
    //   This is why Tensor is a "view", not a "copy".
    // ----------------------------------------------------------

    // ----------------------------------------------------------
    // 3. Querying Tensor properties
    // ----------------------------------------------------------
    printf("\n\n========== 3. Tensor Properties ==========\n");

    printf("\ntensor_col:\n");
    printf("  size(tensor)   = %d  (total element count)\n", int(size(tensor_col)));
    printf("  size<0>(tensor)= %d  (dim-0, number of rows)\n", int(size<0>(tensor_col)));
    printf("  size<1>(tensor)= %d  (dim-1, number of cols)\n", int(size<1>(tensor_col)));
    printf("  rank(tensor)   = %d  (number of dimensions)\n", int(rank(tensor_col)));

    printf("\n  shape  = "); print(shape(tensor_col));  printf("\n");
    printf("  stride = "); print(stride(tensor_col)); printf("\n");
    printf("  layout = "); print(layout(tensor_col)); printf("\n");

    // ----------------------------------------------------------
    // 4. Slicing — extracting a row or column
    // ----------------------------------------------------------
    //
    // One of CuTe's most powerful features:
    //   Extract a row or column from a 2D tensor as a 1D tensor.
    //   No data copy — just a different layout view.
    //
    // Syntax: tensor(m, _)  — extract row m
    //         tensor(_, n)  — extract column n
    //
    // _ is CuTe's wildcard: "keep this entire dimension"
    //

    printf("\n\n========== 4. Slicing ==========\n");

    // --- Extract a column from col-major tensor ---
    // tensor_col layout: (4,3):(1,4)
    // Column 1 = tensor_col(_, 1)
    // Result: 1D tensor, shape=4, stride=1, base offset = 1*4 = 4
    auto col1 = tensor_col(_, 1);  // _ = wildcard
    print_tensor_info("col1 = tensor_col(_, 1)", col1);
    print_tensor_1d("col1", col1);
    // Expected: 4.0  5.0  6.0  7.0

    // --- Extract a row from col-major tensor ---
    // Row 2 = tensor_col(2, _)
    // Result: 1D tensor, shape=3, stride=4
    auto row2 = tensor_col(2, _);
    print_tensor_info("row2 = tensor_col(2, _)", row2);
    print_tensor_1d("row2", row2);
    // Expected: 2.0  6.0  10.0
    // (stride=4: skipping 4 elements between values — crossing columns)

    // --- Extract a row from row-major tensor ---
    auto row1_rm = tensor_row(1, _);
    print_tensor_info("row1_rm = tensor_row(1, _)", row1_rm);
    print_tensor_1d("row1_rm", row1_rm);
    // Expected: 3.0  4.0  5.0
    // (stride=1: contiguous access — rows are contiguous in row-major)

    // --- Extract a column from row-major tensor ---
    auto col0_rm = tensor_row(_, 0);
    print_tensor_info("col0_rm = tensor_row(_, 0)", col0_rm);
    print_tensor_1d("col0_rm", col0_rm);
    // Expected: 0.0  3.0  6.0  9.0
    // (stride=3: jumping 3 elements — columns are strided in row-major)

    // ----------------------------------------------------------
    // Key comparison:
    //   col-major: extract column -> stride=1 (contiguous, cache-friendly!)
    //   col-major: extract row    -> stride=M (strided, not ideal)
    //   row-major: extract row    -> stride=1 (contiguous, cache-friendly!)
    //   row-major: extract column -> stride=N (strided, not ideal)
    //
    //   This is exactly why layout affects performance:
    //   wrong layout = what you think is contiguous is actually scattered.
    // ----------------------------------------------------------

    // ----------------------------------------------------------
    // 5. local_tile — partitioning a Tensor into tiles (intro)
    // ----------------------------------------------------------
    //
    // local_tile is one of CuTe's core tiling utilities.
    //
    // Idea:
    //   Given an (M, N) matrix, split it into (BM, BN) tiles.
    //   local_tile does this for you.
    //
    // Result is a higher-rank tensor:
    //   Original: (M, N)
    //   Tiled:    (BM, BN, num_tiles_m, num_tiles_n)
    //
    // Critical for GPU programming:
    //   each thread block processes one tile.
    //

    printf("\n\n========== 5. local_tile intro ==========\n");

    // Create an 8x6 tensor
    const int M2 = 8, N2 = 6;
    float* data2 = (float*)malloc(M2 * N2 * sizeof(float));
    for (int i = 0; i < M2 * N2; ++i) data2[i] = float(i);

    auto big_tensor = make_tensor(data2, make_shape(M2, N2));
    print_tensor_info("big_tensor (8x6)", big_tensor);
    print_tensor_2d("big_tensor", big_tensor);

    // Tile into 4x3 blocks
    auto tile_shape = make_shape(Int<4>{}, Int<3>{});

    // local_tile result shape: (4, 3, 2, 2)
    //   first two dims:  intra-tile coordinates (4x3)
    //   last two dims:   tile indices (2x2, since 8/4=2, 6/3=2)
    auto tiled = local_tile(big_tensor, tile_shape, make_coord(_, _));

    printf("\nTiled tensor after local_tile:\n");
    printf("  Shape  : "); print(tiled.shape());  printf("\n");
    printf("  Stride : "); print(tiled.stride()); printf("\n");

    // Access tile (0,0) — top-left
    auto tile_00 = tiled(_, _, 0, 0);
    print_tensor_info("tile (0,0)", tile_00);
    print_tensor_2d("tile (0,0)", tile_00);

    // Access tile (1,0) — bottom-left
    auto tile_10 = tiled(_, _, 1, 0);
    print_tensor_info("tile (1,0)", tile_10);
    print_tensor_2d("tile (1,0)", tile_10);

    // Access tile (0,1) — top-right
    auto tile_01 = tiled(_, _, 0, 1);
    print_tensor_info("tile (0,1)", tile_01);
    print_tensor_2d("tile (0,1)", tile_01);

    // Access tile (1,1) — bottom-right
    auto tile_11 = tiled(_, _, 1, 1);
    print_tensor_info("tile (1,1)", tile_11);
    print_tensor_2d("tile (1,1)", tile_11);

    // ----------------------------------------------------------
    // Observations:
    //   tile (0,0) = top-left 4x3 block of big_tensor
    //   tile (1,0) = bottom-left 4x3 block
    //   tile (0,1) = top-right 4x3 block
    //   tile (1,1) = bottom-right 4x3 block
    //
    //   All tiles share the underlying data! Modifying a tile
    //   element will change big_tensor at the same position.
    //
    //   This is the essence of CuTe tiling:
    //     no data copy, just reorganized layout.
    // ----------------------------------------------------------

    // ----------------------------------------------------------
    // 6. Summary
    // ----------------------------------------------------------
    printf("\n============================================================\n");
    printf("  Summary\n");
    printf("============================================================\n");
    printf("\n");
    printf("  Tensor = data pointer + Layout\n");
    printf("\n");
    printf("  Create:  make_tensor(ptr, layout)\n");
    printf("  Access:  tensor(m, n)  =>  *(ptr + layout(m,n))\n");
    printf("  Slice:   tensor(_, n)  =>  column n as 1D tensor\n");
    printf("           tensor(m, _)  =>  row m as 1D tensor\n");
    printf("  Tile:    local_tile(tensor, tile_shape, coord)\n");
    printf("\n");
    printf("  Key: Tensor is a VIEW, not a COPY!\n");
    printf("       Multiple tensors can point to the same memory.\n");
    printf("       Changing layout = changing how you see data, not data itself.\n");
    printf("\n");
    printf("  Next: Partition — distributing Tensor across threads\n");
    printf("============================================================\n");

    free(raw_data);
    free(data2);
    return 0;
}

// ============================================================
//  Concept Reference:
// ============================================================
//
//  CuTe                         Familiar equivalent
//  ---------------------------  ----------------------------------
//  make_tensor(ptr, layout)     Attach an indexing rule to memory
//  tensor(m, n)                 *(ptr + m*stride_m + n*stride_n)
//  tensor(_, n)                 Column n as a 1D view
//  tensor(m, _)                 Row m as a 1D view
//  local_tile(t, shape, coord)  Split a matrix into tiles
//  size(tensor)                 Total element count
//  shape(tensor)                Dimensions of each mode
//  stride(tensor)               Step sizes of each mode
//  layout(tensor)               Complete layout object
//
//  Tensor is a view:            No allocation, no copy — just a lens
//  Multiple Tensors can share:  Like multiple pointers to one malloc
//
