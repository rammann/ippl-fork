#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "OrthoTree/OrthoTree.h"
#include "OrthoTree/OrthoTreeTypes.h"
#include "OrthoTree/BoundingBox.h"

#include <random>

int main(int argc, char* argv[])
{
    ippl::initialize(argc, argv);
    {
        static constexpr size_t Dim = 3;
        const size_t max_particles = 100;
        const size_t max_depth = ;
        const size_t num_particles = 1000;

        const auto MIN_BOUND = 0.0;
        const auto MAX_BOUND = 1.0;

        ippl::OrthoTree<Dim> tree(max_depth, max_particles, ippl::BoundingBox<Dim>({ MIN_BOUND, MIN_BOUND, MIN_BOUND }, { MAX_BOUND, MAX_BOUND, MAX_BOUND }));

        typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
        playout_type PLayout;
        ippl::OrthoTreeParticle particles(PLayout);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(MIN_BOUND, MAX_BOUND);

        particles.create(num_particles);
        for ( unsigned int i = 0; i < num_particles; ++i ) {
            particles.R(i) = ippl::Vector<double, Dim> { unif(eng), unif(eng), unif(eng) };
            particles.rho(i) = 0;
        }

        //tree.build_tree_naive_sequential(particles);
        tree.build_tree_naive(particles);
        std::cout << "working hehe\n";
    }

    ippl::finalize();

    return 0;
}
