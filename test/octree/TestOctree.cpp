#include "Ippl.h"

#include <random>

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef ippl::OctreeParticle<playout_type> particle_type;
        
        int n = 50;

        playout_type playout;
        particle_type particles(playout);
        particles.create(n);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(0, 1);

        for (int i = 0; i < n; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            particles.R(i)                 = r;
            particles.rho(i)               = 0.0;
        }

    }

    ippl::finalize();
    
    return 0;
}