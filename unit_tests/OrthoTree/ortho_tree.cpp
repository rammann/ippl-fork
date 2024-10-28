//
// unit test of morton_codes.h

#include "gtest/gtest.h"

#include "OrthoTree/OrthoTree.h"

#include <algorithm>
#include <cstdint>
#include <bitset>
#include <random>

using namespace ippl;

TEST(OrthoTreeTest, TestTest)
{
    static constexpr size_t Dim = 3;
    OrthoTree<Dim> tree(5, 10, BoundingBox<Dim>({ 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }));

    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    playout_type PLayout;
    OrthoTreeParticle particles(PLayout);

    std::mt19937_64 eng;
    std::uniform_real_distribution<double> unif(0.25, 0.5);

    unsigned int n = 1000;
    particles.create(n);
    for ( unsigned int i = 0; i<n; ++i ) {
        particles.R(i) = Vector<double, 3> { unif(eng),unif(eng),unif(eng) };
        particles.rho(i) = 0;
    }

    tree.build_tree_naive_sequential(particles);

    EXPECT_EQ(true, true);
}
