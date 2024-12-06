#include <Particle/ParticleSpatialLayout.h>
#include <ostream>
#include <random>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "OrthoTree/helpers/AidList.h"
#include "gtest/gtest.h"

using namespace ippl;

template <class PLayout = ippl::ParticleSpatialLayout<double, 3>>
class TestParticle : public ippl::ParticleBase<PLayout> {
public:
    TestParticle() noexcept = default;
    TestParticle(PLayout& L)
        : ippl::ParticleBase<PLayout>(L) {
        ;
    }
};

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
auto getUngatheredParticles(size_t num_particles_per_proc, const double min_bound,
                            double max_bound) {
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
    return bunch;
}

// ================================================================
//                          TESTS START
// ================================================================

TEST(AidListTest, ChecksIfGatheredCorrectly) {
    static constexpr size_t Dim = 3;
    const size_t max_depth      = 5;
    AidList aid_list(max_depth);

    auto gathered_particles   = getParticles<Dim>(100, 0.0, 1.0);
    auto ungathered_particles = getUngatheredParticles<Dim>(100, 0.0, 1.0);

    if (Comm->rank() == 0) {
        EXPECT_FALSE(aid_list.is_gathered(ungathered_particles));
        EXPECT_TRUE(aid_list.is_gathered(gathered_particles));
    } else {
        EXPECT_FALSE(aid_list.is_gathered(ungathered_particles));
        EXPECT_FALSE(aid_list.is_gathered(gathered_particles));
    }
}

TEST(AidListTest, ConstructionTest) {
    static constexpr size_t Dim       = 3;
    const size_t max_depth            = 5;
    const double min_bounds           = 0.0;
    const double max_bounds           = 1.0;
    const size_t n_particles_per_proc = 100;

    BoundingBox<Dim> root_bounds({min_bounds, min_bounds, min_bounds},
                                 {max_bounds, max_bounds, max_bounds});

    auto gathered_particles = getParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);

    AidList working_aid_list(max_depth);
    working_aid_list.initialize<Dim>(root_bounds, gathered_particles);

    if (Comm->rank() == 0) {
        EXPECT_EQ(working_aid_list.size(), n_particles_per_proc * Comm->size());
    } else {
        EXPECT_EQ(working_aid_list.size(), 0);
    }

    auto ungathered_particles =
        getUngatheredParticles<Dim>(n_particles_per_proc, min_bounds, max_bounds);

    AidList failing_aid_list(max_depth);

    if (Comm->rank() == 0) {
        EXPECT_ANY_THROW(failing_aid_list.initialize<Dim>(root_bounds, ungathered_particles));
    } else {
        EXPECT_NO_THROW(failing_aid_list.initialize<Dim>(root_bounds, ungathered_particles));
        EXPECT_EQ(failing_aid_list.size(), 0);
    }
}

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