#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "OrthoTree_old/OrthoTreeParticle.h"
#include "OrthoTree_old/OrthoTree.h"

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {

        // OCTREE CONSTRUCTION TEST 
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;
        ippl::OrthoTreeParticle particles(PLayout);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0.25, 0.5);

        unsigned int n=1000;
        particles.create(n);
        for(unsigned int i=0; i<n; ++i){
            particles.R(i) = ippl::Vector<double, 3>{unif(eng),unif(eng),unif(eng)};
            particles.rho(i) = 0;
        }

        ippl::OrthoTree tree(particles,  2 , 10, 20, ippl::BoundingBox<3>{{0,0,0},{1,1,1}});
        tree.PrintStructure();
    }

    ippl::finalize();

    return 0;
}