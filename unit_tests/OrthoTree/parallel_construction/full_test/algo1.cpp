#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <ostream>
#include <random>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"
#include "test_helper.h"

using namespace ippl;

// ================================================================
//                          TESTS START
// ================================================================

// sanity check, tests are designed to run (and pass) with 4 ranks
TEST(ParallelConstruction, AssertWorldSize) {
    ASSERT_EQ(Comm->size(), 4);
}

TEST(ParallelConstruction, ActualTest) {
    constexpr size_t Dim    = 2;
    const double min_bounds = 0.0;
    const double max_bounds = 1.0;

    const size_t num_particles_per_proc   = 1000;
    const size_t max_particles_per_octant = 100;
    const size_t max_depth                = 8;

    testFunction<Dim>(min_bounds, max_bounds, num_particles_per_proc, max_particles_per_octant,
                      max_depth);
}

// this is required to test the orthotree, as it depends on ippl
int main(int argc, char** argv) {
    // Initialize MPI and IPPL
    ippl::initialize(argc, argv, MPI_COMM_WORLD);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize IPPL and MPI
    ippl::finalize();

    return result;
}
