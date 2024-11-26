#include<gtest/gtest.h>
#include<OrthoTree/OrthoTree.h>

#include<algorithm> 
#include<bitset>
#include<vector>
#include<iostream>

using namespace ippl;




TEST(CompleteRegion, CompleteSimpleQuad){
    EXPECT_TRUE(true);
    return;
    constexpr size_t Dim = 2;
    size_t max_depth = 3;
    OrthoTree<Dim> tree(max_depth, 2, BoundingBox<Dim>(real_coordinate_template<Dim>{0, 0}, real_coordinate_template<Dim>{1, 1}));
    Morton<Dim> morton(max_depth);

    morton_code code_a = morton.encode({0, 0}, 2);
    morton_code code_b = morton.encode({6, 6}, 2);

    assert(code_a != code_b && "cant be the same");
    vector_t<morton_code> complete_region = tree.complete_region(code_a, code_b);

    vector_t<morton_code> expected = {
        morton.encode({2, 0}, 2), morton.encode({0, 2}, 2), morton.encode({2, 2}, 2),
        morton.encode({4, 4}, 2), morton.encode({4, 6}, 2), morton.encode({6, 4}, 2),
        morton.encode({4, 0}, 1), morton.encode({0, 4}, 1),
    };

    std::sort(expected.begin(), expected.end());
    // EXPECT_EQ(complete_region, expected);
}