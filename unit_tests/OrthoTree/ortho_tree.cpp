//
// unit test of morton_codes.h

#include "gtest/gtest.h"

#include "OrthoTree/OrthoTree.h"
#include "OrthoTree/Types.h"

#include <algorithm>
#include <cstdint>
#include <bitset>
#include <vector>
#include <array>

using namespace ippl;

TEST(OrthoTreeTest, Constructor)
{
    static constexpr size_t Dim = 3;
    ippl::OrthoTree<Dim> tree(5, 10, ippl::BoundingBox<Dim>({ 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }));
    EXPECT_EQ(1, 1);
}


TEST(OrthoTreeTest, BuildSimpleQuadTree)
{
    static constexpr size_t Dim = 2;
    ippl::OrthoTree<Dim> tree_2d(5, 1, ippl::BoundingBox<Dim>({ 0.0, 0.0}, { 1.0, 1.0})); 
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    playout_type PLayout;

    OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> particles(PLayout);

    // Example coordinate list for particles
    std::vector<ippl::Vector<double,Dim>> coordinates{{0,0},{0.1,0.6}};

    particles.create(coordinates.size());

    for(size_t i = 0; i < coordinates.size(); i++){
        particles.R(i) = coordinates.at(i);
    }


    ippl::vector_t<morton_code> tree_codes = tree_2d.build_tree_topdown_sequential(0,particles);

    Morton<Dim> morton_helper(5);
    ippl::vector_t<morton_code> expected = morton_helper.get_children(0);
    std::sort(tree_codes.begin(),tree_codes.end());
    EXPECT_EQ(tree_codes, expected);
}

int main(int argc, char** argv) {
    // Initialize MPI and IPPL
    ippl::initialize(argc, argv);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize IPPL and MPI
    ippl::finalize();

    return result;
}
