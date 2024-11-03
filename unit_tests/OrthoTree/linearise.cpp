#include<gtest/gtest.h>
#include "OrthoTree/OrthoTree.h"

using namespace ippl;

TEST(LineariseTestSimple, LineariseOctantsTest) {
    static constexpr size_t Dim = 3;
    const size_t max_depth = 8;
    Morton<Dim> morton(max_depth);
    OrthoTree<Dim> tree(max_depth, 10, BoundingBox<Dim>(real_coordinate_template<Dim>(0), real_coordinate_template<Dim>(1)));

    vector_t<morton_code> octants= { 0b000, 0b001};
    vector_t<morton_code> linearised = tree.linearise_octants(octants);
    vector_t<morton_code> expected = {0b001};
    EXPECT_EQ(linearised, expected);
}