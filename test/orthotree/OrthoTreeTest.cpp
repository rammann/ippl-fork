#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "OrthoTree/OrthoTree.h"
#include "OrthoTree/Types.h"
#include "OrthoTree/BoundingBox.h"

#include <random>

int main(int argc, char* argv[])
{
    ippl::initialize(argc, argv);
    {
        static constexpr size_t Dim = 3;
        ippl::OrthoTree<Dim> tree(5, 10, ippl::BoundingBox<Dim>({ 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 }));

        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;
        ippl::OrthoTreeParticle particles(PLayout);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0.25, 0.5);

        unsigned int n = 100;
        particles.create(n);
        for ( unsigned int i = 0; i<n; ++i ) {
            particles.R(i) = ippl::Vector<double, 3> { unif(eng),unif(eng),unif(eng) };
            particles.rho(i) = 0;
        }

        tree.build_tree_naive_sequential(particles);
        std::cout << "working hehe\n";
    }

    ippl::finalize();

    return 0;
}
