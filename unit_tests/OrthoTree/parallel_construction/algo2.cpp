#include<gtest/gtest.h>
#include<OrthoTree/OrthoTree.h>

#include<algorithm> 
#include<bitset>
#include<vector>
#include<iostream>

using namespace ippl;




TEST(CompleteRegion, CompleteSimpleQuad){
    // EXPECT_TRUE(true);
    // return;
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    morton_code code_a = morton.encode({0, 0}, 2);
    morton_code code_b = morton.encode({6, 6}, 2);

    assert(code_a != code_b && "cant be the same");
    Kokkos::View<morton_code*> complete_region = tree.complete_region(code_a, code_b);

    size_t num_elements = 8;
    Kokkos::View<morton_code*> expected("expected", num_elements);
    expected(0) = morton.encode({2, 0}, 2);
    expected(1) = morton.encode({0, 2}, 2);
    expected(2) = morton.encode({2, 2}, 2);
    expected(3) = morton.encode({4, 4}, 2);
    expected(4) = morton.encode({4, 6}, 2);
    expected(5) = morton.encode({6, 4}, 2);
    expected(6) = morton.encode({4, 0}, 1);
    expected(7) = morton.encode({0, 4}, 1);

    Kokkos::parallel_sort(expected);
    EXPECT_EQ(complete_region, expected);
}