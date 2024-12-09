#pragma once

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <exception>
#include <gtest/gtest.h>
#include <ostream>
#include <random>
#include <sstream>
#include <stdexcept>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"

using namespace ippl;

template <size_t Dim>
std::string generateCommandToReplicate(int world_size, uint64_t seed, size_t num_particles_per_proc,
                                       size_t max_depth, size_t max_particles_per_octant,
                                       double min_bounds, double max_bounds) {
    std::ostringstream command;
    command << std::endl
            << "./visualise " << world_size << " dim=" << Dim << " seed=" << seed
            << " num_particles=" << num_particles_per_proc << " max_depth=" << max_depth
            << " max_particles=" << max_particles_per_octant << " min_bounds=" << min_bounds
            << " max_bounds=" << max_bounds << std::endl;
    return command.str();
}

template <size_t Dim>
auto getParticles(size_t num_particles_per_proc, const double min_bound, double max_bound,
                  uint64_t seed) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bound, max_bound);

    bunch.create(num_particles_per_proc);

    typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();
    for (unsigned int i = 0; i < num_particles_per_proc; ++i) {
        if constexpr (Dim == 3) {
            R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng), unif(eng)};
        } else if constexpr (Dim == 2) {
            R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng)};
        }
    }

    Kokkos::deep_copy(bunch.R.getView(), R_host);
    bunch.update();
    return bunch;
}

template <size_t Dim>
void testRun(double min_bounds, double max_bounds, size_t max_particles_per_octant,
             size_t max_depth, const auto& particles, const std::string& command_to_replicate) {
    BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});
    OrthoTree<Dim> tree(max_depth, max_particles_per_octant, root_bounds);

    Kokkos::View<ippl::morton_code*> parallel_tree;
    Kokkos::View<ippl::morton_code*> sequential_tree;

    parallel_tree = tree.build_tree(particles);
    // sequential_tree = tree.build_tree_naive(particles);

    const size_t world_rank = static_cast<size_t>(Comm->rank());
    const size_t world_size = static_cast<size_t>(Comm->size());

    // here we gather all octants on rank 0 s.t. we can compare them to the naive implementation
    size_t local_size = parallel_tree.extent(0);
    std::vector<size_t> local_sizes(world_size);
    Comm->gather(&local_size, local_sizes.data(), 1);

    if (world_rank == 0) {
        const size_t total_size = std::accumulate(local_sizes.begin(), local_sizes.end(), 0);
        Kokkos::resize(parallel_tree, total_size);
        size_t offset = 0;
        for (size_t rank = 1; rank < world_size; ++rank) {
            offset += local_sizes[rank - 1];
            mpi::Status data_status;
            Comm->recv(parallel_tree.data() + offset, local_sizes[rank], rank, 0, data_status);
        }
    } else {
        Comm->send(*parallel_tree.data(), local_size, 0, 0);
    }

    if (world_rank == 0) {
        // sorting to be sure
        std::sort(parallel_tree.data(), parallel_tree.data() + parallel_tree.extent(0));
        std::sort(sequential_tree.data(), sequential_tree.data() + sequential_tree.extent(0));

        // asserting total size matches
        const size_t total_size    = parallel_tree.extent(0);
        const size_t expected_size = sequential_tree.extent(0);
        ASSERT_EQ(total_size, expected_size) << "Replicate with: " << command_to_replicate;

        // asserting each octant
        for (size_t i = 0; i < total_size; ++i) {
            morton_code par_octant = parallel_tree[i];
            morton_code seq_octant = sequential_tree[i];
            ASSERT_EQ(par_octant, seq_octant) << "Replicate with : " << command_to_replicate;
        }
    }
}

/**
 * @brief We call this function from multiple test files, each running with a different number
 * of ranks and each one running all those tests.
 */
void runTests() {
    /**
     * @brief This is a collection of test data. This way we can test and compare the
     * construction of our tree with the sequential version with multiple different params.
     */
    struct {
        std::vector<double> min_bounds_v = {-1.0, 0.0};
        double max_bounds                = 1.0;  // 2

        std::vector<size_t> num_particles_v       = {100, 500, 1000, 2000};       // 4
        std::vector<size_t> max_depth_v           = {2, 4, 6, 8, 10};             // 5
        std::vector<double> max_particles_ratio_v = {0.01, 0.05, 0.1, 0.2, 0.5};  // 5
    } test_data;  // total = 2 * 4 * 5 * 5 = 200 runs lol

    static constexpr size_t Dim2 = 2;
    static constexpr size_t Dim3 = 3;
    // * 2 = 400 runs
    // -> we test it for world_size in {2, 4, 8} -> 1600 runs whupsi

    const size_t seed       = std::random_device{}();
    const double max_bounds = test_data.max_bounds;

    for (size_t num_particles : test_data.num_particles_v) {
        for (double min_bounds : test_data.min_bounds_v) {
            auto particles2D = getParticles<Dim2>(num_particles, min_bounds, max_bounds, seed);
            auto particles3D = getParticles<Dim3>(num_particles, min_bounds, max_bounds, seed);
            for (size_t max_depth : test_data.max_depth_v) {
                for (double max_particles_ratio : test_data.max_particles_ratio_v) {
                    const size_t max_particles = num_particles * max_particles_ratio;

                    testRun<Dim2>(min_bounds, max_bounds, max_particles, max_depth, particles2D,
                                  generateCommandToReplicate<Dim2>(
                                      Comm->rank(), seed, num_particles, max_depth, max_particles,
                                      min_bounds, max_bounds));
                    testRun<Dim3>(min_bounds, max_bounds, max_particles, max_depth, particles3D,
                                  generateCommandToReplicate<Dim3>(
                                      Comm->rank(), seed, num_particles, max_depth, max_particles,
                                      min_bounds, max_bounds));
                }
            }
        }
    }
}

#undef COMMAND_TO_REPLICATE_RUN