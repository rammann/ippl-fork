#include<gtest/gtest.h>
#include "OrthoTree/OrthoTree.h"

using namespace ippl;

TEST(LineariseTestOct, LineariseOctantsTest) {
    static constexpr size_t Dim = 3;
    const size_t max_depth = 3;
    Morton<Dim> morton(max_depth);
    OrthoTree<Dim> tree(max_depth, 10, BoundingBox<Dim>(real_coordinate_template<Dim>(0), real_coordinate_template<Dim>(1)));

    vector_t<morton_code> octs = {morton.encode({0, 0, 0}, 2), morton.encode({0, 4, 0}, 1), morton.encode({0, 6, 6}, 2)};
    vector_t<morton_code> expected = {};

    size_t n_parents = octs.size();

    for(size_t i = 0; i < n_parents; ++i)
    {
        vector_t<morton_code> children = morton.get_children(octs[i]);
        for(size_t j = 0; j < 8; ++j)
        {
            octs.push_back(children[j]);
            expected.push_back(children[j]);
        }
    }

    //octants need to be sorted
    std::sort(expected.begin(), expected.end());
    std::sort(octs.begin(), octs.end());


    vector_t<morton_code> linearised = tree.linearise_octants(octs);
    EXPECT_EQ(linearised, expected);
}