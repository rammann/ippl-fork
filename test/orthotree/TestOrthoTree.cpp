#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {
        /*  UNORDERED MAP TEST 
        Kokkos::UnorderedMap<int, double> map;
        map.insert(1, 1.5);
        map.insert(2, 2.5);

        double& ref = map.value_at(map.find(1));
        ref = 12.5;
        
        std::cout << map.value_at(map.find(1)) << " " << map.value_at(map.find(2)) << std::endl;
        */

        /* OCTREE CONSTRUCTION TEST */
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;
        ippl::OctreeParticle particles(PLayout, 5);
        particles.create(3);
        particles.R(0) = ippl::Vector<double,3>{0.1,0.1,0.1};
        particles.R(1) = ippl::Vector<double,3>{0.2,0.2,0.2};
        particles.R(2) = ippl::Vector<double,3>{0.3,0.3,0.3};

        ippl::OrthoTree tree(particles, 3, ippl::BoundingBox{{0,0,0},{2,2,2}}, 2);
        

        /* LocationIterator distance TEST 
        
        Kokkos::vector<double> vec(100);
        using LocationIterator = typename Kokkos::vector<double>::iterator;
        LocationIterator begin = vec.begin();
        LocationIterator end = vec.end();

        std::cout << end - begin << std::endl;
        */
    }

    ippl::finalize();
    
    return 0;
}