#include "Ippl.h"
#include <Kokkos_Vector.hpp>

#include <random>

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        typedef ippl::OctreeParticle<playout_type> particle_type;
        typedef ippl::Octree::OctreeContainer<ippl::Octree::Octree> container_type;
        
        
        unsigned int nsources = 50;
        //unsigned int ntargets = 20;
        
        playout_type playout;
        particle_type particles(playout, 4);
        particles.create(nsources);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(-0.5, 0.5);

        for (unsigned int i = 0; i < nsources; ++i) {
            ippl::Vector<double, 3> r = {unif(eng), unif(eng), unif(eng)};
            //std::cout << r[0] << " " << r[1] << " " << r[2] << std::endl;
            particles.R(i)                 = r;
            particles.rho(i)               = 0.0;
        }
        particles.R.getParticleCount();
        auto const bounding_box_m = ippl::Octree::BoundingBox3D{{-0.5,-0.5,-0.5}, {0.5,0.5,0.5}};

        container_type container(particles, 5, bounding_box_m, 3);

        

    }

    ippl::finalize();
    
    return 0;
}