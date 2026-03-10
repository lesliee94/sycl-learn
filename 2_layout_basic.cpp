// ============================================================
// Day 1: CuTe Layout Basics
// ============================================================
//
// A Layout in CuTe is a function: coordinate -> index
// It is defined by two IntTuples: Shape and Stride
//
//   index = sum_i(coord_i * stride_i)
//
// This demo covers:
//   1. Creating 1D layouts (vectors)
//   2. Creating 2D layouts (matrices) — column-major & row-major
//   3. Custom strides and padded layouts
//   4. Hierarchical (nested) shapes
//   5. Static vs dynamic integers
//   6. Querying layout properties: size, rank, shape, stride, depth, cosize
//   7. Indexing with 1D, 2D, and hierarchical coordinates
//   8. Accessing sub-layouts of a hierarchical layout
//   9. print_layout visualization
//
// Compile:
//   icpx -fsycl -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -I ../sycl-tla/include 2_layout_basic.cpp -o 2_layout_basic
//
// ============================================================

#include <cute/tensor.hpp>
#include <cstdio>

using namespace cute;

// Helper: print a rank-2 layout as a 2D table
template <class Shape, class Stride>
void print2D(const char* name, Layout<Shape, Stride> const& layout) {
    printf("\n=== %s ===\n", name);
    printf("Layout: ");
    print(layout);
    printf("\n");

    // Iterate over 2D coordinates and print the mapped index
    for (int m = 0; m < size<0>(layout); ++m) {
        for (int n = 0; n < size<1>(layout); ++n) {
            printf("%3d  ", layout(m, n));
        }
        printf("\n");
    }
}

// Helper: print a layout as a flat 1D sequence
template <class Shape, class Stride>
void print1D(const char* name, Layout<Shape, Stride> const& layout) {
    printf("\n=== %s (1D) ===\n", name);
    printf("Layout: ");
    print(layout);
    printf("\n");

    for (int i = 0; i < size(layout); ++i) {
        printf("%3d  ", layout(i));
    }
    printf("\n");
}

// Helper: print basic metadata of a layout
template <class LayoutT>
void printLayoutInfo(const char* name, LayoutT const& layout) {
    printf("\n=== %s ===\n", name);
    printf("Layout : "); print(layout);        printf("\n");
    printf("Shape  : "); print(shape(layout)); printf("\n");
    printf("Stride : "); print(stride(layout));printf("\n");
    printf("Rank   : %d\n", int(rank(layout)));
    printf("Depth  : %d\n", int(depth(layout)));
    printf("Size   : %d\n", int(size(layout)));
    printf("Cosize : %d\n", int(cosize(layout)));
}

int main() {
    printf("============================================================\n");
    printf("  CuTe Layout Basics - Day 1\n");
    printf("============================================================\n");

    // ----------------------------------------------------------
    // 1. Vector layouts (rank-1)
    // ----------------------------------------------------------
    // Shape 8, Stride 1 => contiguous 8-element vector
    // index = coord * 1
    auto vec_contig = make_layout(Int<8>{});
    print1D("vec_contig: 8:1", vec_contig);

    // Shape 8, Stride 2 => strided vector, every other element
    // index = coord * 2
    auto vec_strided = make_layout(Int<8>{}, Int<2>{});
    print1D("vec_strided: 8:2", vec_strided);

    // ----------------------------------------------------------
    // 2. Matrix layouts (rank-2): column-major vs row-major
    // ----------------------------------------------------------

    // Column-major 4x2: stride-1 down columns, stride-4 across rows
    //   Shape  = (4, 2)
    //   Stride = (1, 4)     <- LayoutLeft (default)
    //
    // Memory:  [0 1 2 3 | 4 5 6 7]
    //            col 0     col 1
    auto col_major = make_layout(make_shape(Int<4>{}, Int<2>{}));
    print2D("col_major (4,2):(1,4)", col_major);

    // Row-major 4x2: stride-2 down columns, stride-1 across rows
    //   Shape  = (4, 2)
    //   Stride = (2, 1)
    auto row_major = make_layout(make_shape(Int<4>{}, Int<2>{}),
                                 LayoutRight{});
    print2D("row_major (4,2):(2,1)", row_major);

    // ----------------------------------------------------------
    // 3. Custom stride: non-standard memory layout
    // ----------------------------------------------------------
    // Shape = (2, 4), Stride = (12, 1)
    // Row 0: indices 0,1,2,3    (stride-1 across columns)
    // Row 1: indices 12,13,14,15 (jump by 12 to next row)
    // Useful for: skipping padding, strided access patterns
    auto custom = make_layout(make_shape(Int<2>{}, 4),
                              make_stride(Int<12>{}, Int<1>{}));
    print2D("custom (2,4):(12,1)", custom);

    // Padded layout: logical size is still 8, but the codomain is 28.
    // This is useful when rows/columns are padded for alignment.
    auto padded = make_layout(make_shape(Int<4>{}, Int<2>{}),
                              make_stride(Int<1>{}, Int<24>{}));
    print2D("padded (4,2):(1,24)", padded);
    printLayoutInfo("padded metadata", padded);

    // ----------------------------------------------------------
    // 4. Hierarchical (nested) shapes
    // ----------------------------------------------------------
    // Shape = (2, (2,2)), Stride = (4, (2,1))
    // Logically 2x4, but the second mode is split into (2,2)
    // This is NOT row-major or column-major!
    auto hier = make_layout(make_shape(2, make_shape(2, 2)),
                            make_stride(4, make_stride(2, 1)));

    print2D("hierarchical (2,(2,2)):(4,(2,1))", hier);
    // Also show it in flat 1D to see the full index pattern
    print1D("hierarchical flat", hier);

    // Compare: same logical shape but with LayoutLeft => col-major
    auto hier_col = make_layout(make_shape(2, make_shape(2, 2)),
                                LayoutLeft{});
    print2D("hier_col (2,(2,2)) LayoutLeft", hier_col);

    // Same logical hierarchical shape, but generate strides from the right.
    // For hierarchical shapes, LayoutRight may look less intuitive.
    auto hier_row = make_layout(shape(hier), LayoutRight{});
    print2D("hier_row (2,(2,2)) LayoutRight", hier_row);

    // ----------------------------------------------------------
    // 5. Static vs dynamic integers
    // ----------------------------------------------------------
    // Static: Int<N>{} or _N{} — known at compile time
    // Dynamic: int/size_t — known only at run time
    //
    // CuTe prints static as _N and dynamic as N
    auto mixed = make_layout(make_shape(Int<2>{}, 4),
                             make_stride(Int<1>{}, Int<2>{}));
    printf("\n=== mixed static/dynamic ===\n");
    printf("Layout: ");
    print(mixed);
    printf("\n");
    // Note: _2 is static, 4 is dynamic in the shape

    // ----------------------------------------------------------
    // 6. Querying layout properties
    // ----------------------------------------------------------
    printf("\n=== Layout properties ===\n");

    printf("col_major:\n");
    printf("  rank   = %d\n", int(rank(col_major)));
    printf("  size   = %d (total elements)\n", int(size(col_major)));
    printf("  size<0>= %d (rows)\n", int(size<0>(col_major)));
    printf("  size<1>= %d (cols)\n", int(size<1>(col_major)));
    printf("  shape  = "); print(shape(col_major));  printf("\n");
    printf("  stride = "); print(stride(col_major)); printf("\n");
    printf("  cosize = %d (codomain size)\n", int(cosize(col_major)));

    printf("\nhier:\n");
    printf("  rank   = %d\n", int(rank(hier)));
    printf("  depth  = %d\n", int(depth(hier)));
    printf("  size   = %d\n", int(size(hier)));
    printf("  size<0>= %d\n", int(size<0>(hier)));
    printf("  size<1>= %d (second mode, flattened 2*2=4)\n", int(size<1>(hier)));
    printf("  shape<0>= "); print(shape<0>(hier)); printf("\n");
    printf("  shape<1>= "); print(shape<1>(hier)); printf("\n");
    printf("  stride<0>= "); print(stride<0>(hier)); printf("\n");
    printf("  stride<1>= "); print(stride<1>(hier)); printf("\n");

    // ----------------------------------------------------------
    // 7. Indexing: 1D vs 2D coordinates
    // ----------------------------------------------------------
    printf("\n=== Indexing demo ===\n");
    printf("col_major(0,0) = %d\n", int(col_major(0, 0)));
    printf("col_major(2,1) = %d  (row=2, col=1 => 2*1 + 1*4 = 6)\n",
           int(col_major(2, 1)));
    printf("col_major(5)   = %d  (1D coord 5 => same as (1,1))\n",
           int(col_major(5)));

    printf("\nrow_major(2,1) = %d  (row=2, col=1 => 2*2 + 1*1 = 5)\n",
           int(row_major(2, 1)));

        // Hierarchical coordinates can be addressed in multiple compatible ways.
        printf("\nhier(7)                     = %d  (flat 1D coordinate)\n", int(hier(7)));
        printf("hier(1,3)                   = %d  (logical 2D coordinate)\n", int(hier(1, 3)));
        printf("hier(make_coord(1, make_coord(1,1))) = %d  (hierarchical coordinate)\n",
            int(hier(make_coord(1, make_coord(1, 1)))));

        // Sub-layouts: get<0> is the first mode, get<1> is the hierarchical second mode.
        auto hier_mode0 = get<0>(hier);
        auto hier_mode1 = get<1>(hier);
        printLayoutInfo("get<0>(hier)", hier_mode0);
        printLayoutInfo("get<1>(hier)", hier_mode1);
        print1D("get<1>(hier) flattened", hier_mode1);

    // ----------------------------------------------------------
        // 8. print_layout: CuTe's built-in formatted 2D visualization
    // ----------------------------------------------------------
    printf("\n=== print_layout (CuTe built-in) ===\n");
    print_layout(col_major);
    printf("\n");
    print_layout(row_major);
    printf("\n");
        print_layout(hier);
        printf("\n");
        print_layout(hier_row);

    return 0;
}
