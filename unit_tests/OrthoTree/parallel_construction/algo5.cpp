#include <ostream>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"

// simple test for partition function.
// we assume this test suite is run with 4 processes
TEST(ParallelOrthoTreeTest, PartitionTestDistribute) {
    ippl::mpi::Communicator comm;
    unsigned rank    = comm.rank();
    unsigned n_procs = comm.size();
    // EXPECT_EQ(n_procs, 4);
    if (n_procs != 4) {
        return;
    }

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
    unsigned rank    = comm.rank();
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
            data[i] = 3 * rank + i;
        }
        result = tree_3d.partition(data, weights);
    } else {
        Kokkos::vector<ippl::morton_code> data;
        weights.clear();
        result = tree_3d.partition(data, weights);
    }

    if (rank == 0 || rank == 2) {
        expected.resize(1);
        expected[0] = 3 * rank / 2;
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

TEST(ParallelOrthoTreeTest, PartitionTestDistributeUneven) {
    ippl::mpi::Communicator comm;
    unsigned rank    = comm.rank();
    unsigned n_procs = comm.size();
    EXPECT_EQ(n_procs, 4);
    ippl::OrthoTree<3> tree_3d(5, 1, ippl::BoundingBox<3>({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}));
    Kokkos::vector<size_t> weights;
    Kokkos::vector<ippl::morton_code> expected;
    Kokkos::vector<ippl::morton_code> result;
    if (rank == 0) {
        Kokkos::vector<ippl::morton_code> data(6, 1);
        weights  = Kokkos::vector<size_t>(6, 1);
        expected = Kokkos::vector<ippl::morton_code>(3, 1);
        result   = tree_3d.partition(data, weights);

    } else if (rank == 1) {
        Kokkos::vector<ippl::morton_code> data(2, 1);
        weights  = Kokkos::vector<size_t>(2, 1);
        expected = Kokkos::vector<ippl::morton_code>(3, 1);
        result   = tree_3d.partition(data, weights);
    } else {
        Kokkos::vector<ippl::morton_code> data(1, 1);
        weights  = Kokkos::vector<size_t>(1, 1);
        expected = Kokkos::vector<ippl::morton_code>(2, 1);
        result   = tree_3d.partition(data, weights);
    }
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); i++) {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(ParallelOrthoTreeTest, GetNumParticles) {
    // parallel stuff
    ippl::mpi::Communicator comm;
    unsigned rank    = comm.rank();
    unsigned n_procs = comm.size();
    EXPECT_EQ(n_procs, 4);

    // octree stuff
    size_t max_depth = 3;

    ippl::OrthoTree<2> tree_2d(max_depth, 1, ippl::BoundingBox<2>({0.0, 0.0}, {1.0, 1.0}));
    ippl::Morton<2> morton_helper(max_depth);

    if (rank == 0) {
        Kokkos::vector<Kokkos::pair<ippl::morton_code, size_t>> aid_list;

        // for rank 0
        aid_list.push_back({morton_helper.encode({0, 0}, 3), 0});
        aid_list.push_back({morton_helper.encode({0, 0}, 3), 0});
        aid_list.push_back({morton_helper.encode({1, 1}, 3), 0});
        aid_list.push_back({morton_helper.encode({3, 3}, 3), 0});

        // for rank 1
        aid_list.push_back({morton_helper.encode({6, 0}, 3), 0});
        aid_list.push_back({morton_helper.encode({6, 1}, 3), 0});
        aid_list.push_back({morton_helper.encode({1, 6}, 3), 0});

        // for rank 3
        aid_list.push_back({morton_helper.encode({7, 7}, 3), 0});
        aid_list.push_back({morton_helper.encode({7, 7}, 3), 0});
        aid_list.push_back({morton_helper.encode({7, 7}, 3), 0});

        tree_2d.set_aid_list(aid_list);

        Kokkos::vector<ippl::morton_code> octants;
        octants.push_back(morton_helper.encode({0, 0}, 1));

        // test for rank 0
        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);
        Kokkos::vector<size_t> expected(1);
        expected[0] = 4;

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }

    if (rank == 1) {
        Kokkos::vector<ippl::morton_code> octants;

        octants.push_back(morton_helper.encode({6, 0}, 2));
        octants.push_back(morton_helper.encode({4, 2}, 2));
        octants.push_back(morton_helper.encode({0, 6}, 2));

        Kokkos::vector<size_t> expected(3);
        expected[0] = 2;
        expected[1] = 0;
        expected[2] = 1;

        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }

    if (rank == 2) {
        Kokkos::vector<ippl::morton_code> octants;

        octants.push_back(morton_helper.encode({2, 4}, 2));

        Kokkos::vector<size_t> expected(1);
        expected[0] = 0;

        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }

    if (rank == 3) {
        Kokkos::vector<ippl::morton_code> octants;

        octants.push_back(morton_helper.encode({6, 4}, 2));
        octants.push_back(morton_helper.encode({7, 7}, 3));

        Kokkos::vector<size_t> expected(2);
        expected[0] = 0;
        expected[1] = 3;

        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }
}
TEST(ParallelOrthoTreeTest, GetNumParticlesProcessorEmpty) {
    // parallel stuff
    ippl::mpi::Communicator comm;
    unsigned rank    = comm.rank();
    unsigned n_procs = comm.size();
    EXPECT_EQ(n_procs, 4);

    // octree stuff
    size_t max_depth = 3;

    ippl::OrthoTree<2> tree_2d(max_depth, 1, ippl::BoundingBox<2>({0.0, 0.0}, {1.0, 1.0}));
    ippl::Morton<2> morton_helper(max_depth);

    if (rank == 0) {
        Kokkos::vector<Kokkos::pair<ippl::morton_code, size_t>> aid_list;

        // for rank 0
        aid_list.push_back({morton_helper.encode({0, 0}, 3), 0});
        aid_list.push_back({morton_helper.encode({0, 0}, 3), 0});
        aid_list.push_back({morton_helper.encode({1, 1}, 3), 0});
        aid_list.push_back({morton_helper.encode({3, 3}, 3), 0});

        // for rank 1
        aid_list.push_back({morton_helper.encode({6, 0}, 3), 0});
        aid_list.push_back({morton_helper.encode({6, 1}, 3), 0});
        aid_list.push_back({morton_helper.encode({1, 6}, 3), 0});

        // for rank 3
        aid_list.push_back({morton_helper.encode({7, 7}, 3), 0});
        aid_list.push_back({morton_helper.encode({7, 7}, 3), 0});
        aid_list.push_back({morton_helper.encode({7, 7}, 3), 0});

        tree_2d.set_aid_list(aid_list);

        Kokkos::vector<ippl::morton_code> octants;
        octants.push_back(morton_helper.encode({0, 0}, 1));

        // test for rank 0
        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);
        Kokkos::vector<size_t> expected(1);
        expected[0] = 4;

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }

    if (rank == 1) {
        Kokkos::vector<ippl::morton_code> octants;

        Kokkos::vector<size_t> expected;

        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }

    if (rank == 2) {
        Kokkos::vector<ippl::morton_code> octants;

        octants.push_back(morton_helper.encode({2, 4}, 2));
        Kokkos::vector<size_t> expected(1);
        expected[0] = 0;

        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }

    if (rank == 3) {
        Kokkos::vector<ippl::morton_code> octants;

        octants.push_back(morton_helper.encode({6, 4}, 2));
        octants.push_back(morton_helper.encode({7, 7}, 3));

        Kokkos::vector<size_t> expected(2);
        expected[0] = 0;
        expected[1] = 3;

        Kokkos::vector<size_t> result = tree_2d.get_num_particles_in_octants_parallel(octants);

        ASSERT_EQ(result.size(), expected.size());
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(result[i], expected[i]);
        }
    }
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
