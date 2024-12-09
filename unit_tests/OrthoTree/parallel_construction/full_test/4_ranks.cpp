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
TEST(ParallelConstruction4, AssertWorldSize) {
    ASSERT_EQ(Comm->size(), 4);
}

TEST(ParallelConstruction4, ActualTest) {
    runTests();
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
