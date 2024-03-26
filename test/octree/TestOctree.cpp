#include "Ippl.h"

#include <random>

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef ippl::OctreeParticle<playout_type> particle_type;
        //typedef ippl::OrthoTree::OrthoTreeContainer<ippl::OrthoTree::OrthoTreePoint<3,> treecontainer_type;
        
        unsigned int nsources = 50;
        unsigned int ntargets = 20;
        

        playout_type playout;
        particle_type particles(playout, 50);
        particles.create(nsources + ntargets);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(-0.5, 0.5);

        for (unsigned int i = 0; i < nsources + ntargets; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            particles.R(i)                 = r;
            particles.rho(i)               = 0.0;
        }

        

    }

    ippl::finalize();
    
    return 0;
}