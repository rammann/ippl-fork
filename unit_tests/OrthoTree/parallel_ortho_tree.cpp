#include <ostream>
#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"


#include "OrthoTree/OrthoTreeTypes.h"

// simple test for partition function.
// we assume this test suite is run with 4 processes
TEST(ParallelOrthoTreeTest, PartitionTestDistribute) {
  ippl::mpi::Communicator comm;
  unsigned rank = comm.rank();
  unsigned n_procs = comm.size();
  EXPECT_EQ(n_procs, 4);
  ippl::OrthoTree<3> tree_3d(5, 1, ippl::BoundingBox<3>({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}));
  Kokkos::vector<size_t> weights(8, 1);
  Kokkos::vector<ippl::morton_code> expected(2, 1);
  Kokkos::vector<ippl::morton_code> result;
  if (rank == 0) {
    Kokkos::vector<ippl::morton_code> data(8, 1);
    result = tree_3d.partition(data, weights);
  } else {
    Kokkos::vector<ippl::morton_code> data;
    weights.clear();
    result = tree_3d.partition(data, weights);
  }
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(result[i], expected[i]); 
  }
}

TEST(ParallelOrthoTreeTest, PartitionTestWeighted) {
  ippl::mpi::Communicator comm;
  unsigned rank = comm.rank();
  unsigned n_procs = comm.size();
  EXPECT_EQ(n_procs, 4);
  ippl::OrthoTree<3> tree_3d(5, 1, ippl::BoundingBox<3>({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}));
  Kokkos::vector<size_t> weights(3, 1);
  weights[0] = 2;
  Kokkos::vector<ippl::morton_code> expected(2, 1);
  Kokkos::vector<ippl::morton_code> result;
  if (rank == 0 || rank == 1) {
    Kokkos::vector<ippl::morton_code> data(3);
    for (unsigned int i = 0; i < data.size(); i++) {
      data[i] = 3*rank + i; 
    }
    result = tree_3d.partition(data, weights);
  } else {
    Kokkos::vector<ippl::morton_code> data;
    weights.clear();
    result = tree_3d.partition(data, weights);
  }

  if (rank == 0 || rank == 2) {
    expected.resize(1);
    expected[0] = 3*rank/2;
  } else if (rank == 1) {
    expected.resize(2);
    expected[0] = 1;
    expected[1] = 2;
  } else {
    expected[0] = 4;
    expected[1] = 5;
  }
  ASSERT_EQ(result.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_EQ(result[i], expected[i]); 
  }
} 

// this is required to test the orthotree, as it depends on ippl
int main(int argc, char** argv)
{
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
