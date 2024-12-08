#pragma once

#include <Kokkos_Core.hpp>

#include <Kokkos_Sort.hpp>
#include <ostream>
#include <random>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/OrthoTree.h"
#include "gtest/gtest.h"

using namespace ippl;

template <size_t Dim>
auto getParticles(size_t num_particles_per_proc, const double min_bound, double max_bound) {
    typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
    typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

    playout_type PLayout;
    bunch_type bunch(PLayout);

    std::mt19937_64 eng;
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
void testFunction(double min_bounds, double max_bounds, size_t num_particles_per_proc,
                  size_t max_particles_per_octant, size_t max_depth) {
    const size_t world_rank = static_cast<size_t>(Comm->rank());
    const size_t world_size = static_cast<size_t>(Comm->size());

    BoundingBox<Dim> root_bounds({min_bounds, min_bounds}, {max_bounds, max_bounds});
    OrthoTree<Dim> tree(max_depth, max_particles_per_octant, root_bounds);

    auto particles       = getParticles<Dim>(num_particles_per_proc, min_bounds, max_bounds);
    auto parallel_tree   = tree.build_tree(particles);        // Kokkos::View<size_t*>
    auto sequential_tree = tree.build_tree_naive(particles);  // Kokkos::View<size_t*>

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

#define OUR_DEBUG_ERROR_MESSAGE                                                           \
    "Failed for world_size=" << world_size << ", dim=" << Dim                             \
                             << ", n_particles=" << num_particles_per_proc                \
                             << ", max_depth=" << max_depth                               \
                             << ", max_particles_per_octant=" << max_particles_per_octant \
                             << ", min_bounds=" << min_bounds << ", max_bounds=" << max_bounds

    if (world_rank == 0) {
        // sorting to be sure
        std::sort(parallel_tree.data(), parallel_tree.data() + parallel_tree.extent(0));
        std::sort(sequential_tree.data(), sequential_tree.data() + sequential_tree.extent(0));

        // asserting total size matches
        const size_t total_size = parallel_tree.extent(0);
        ASSERT_EQ(total_size, sequential_tree.extent(0)) << OUR_DEBUG_ERROR_MESSAGE;

        // asserting each octant
        for (size_t i = 0; i < total_size; ++i) {
            morton_code par_octant = parallel_tree[i];
            morton_code seq_octant = sequential_tree[i];
            ASSERT_EQ(par_octant, seq_octant) << OUR_DEBUG_ERROR_MESSAGE;
        }
    }

#undef OUR_DEBUG_ERROR_MESSAGE
}