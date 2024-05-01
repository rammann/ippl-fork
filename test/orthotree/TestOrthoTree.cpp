#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"

int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    
    {

        /* Plotting Test */




        /* SOLVER INIT TEST*/
        
        // Particle layout type
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;

        // Targets
        ippl::OrthoTreeParticle targets(PLayout);
        unsigned int nTargets = 200;
        targets.create(nTargets);

        // Sources
        ippl::OrthoTreeParticle sources(PLayout);
        unsigned int nSources = 200;
        sources.create(nSources);

        // Random generators for position and charge
        std::mt19937_64 eng(42);
        std::uniform_real_distribution<double> posDis(0.0, 1.0);
        std::uniform_real_distribution<double> chargeDis(-20,20);

        // Generate target points
        
        for(unsigned int idx=0; idx<nTargets; ++idx){
            ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
            targets.R(idx) = r;
            targets.rho(idx) = 0.0;
        }

        // Generate source points
        for(unsigned int idx=0; idx<nSources; ++idx){
            ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
            sources.R(idx) = r;
            sources.rho(idx) = chargeDis(eng);
        }
       
       
       



        // Tree Params
        ippl::ParameterList treeparams;
        treeparams.add("maxdepth",          7);
        treeparams.add("maxleafelements",   20);
        treeparams.add("boxmin",            0.0);
        treeparams.add("boxmax",            1.0);
        treeparams.add("sourceidx",         nTargets);

        // Solver Params
        ippl::ParameterList solverparams;
        solverparams.add("eps", 0.000001);

        
        ippl::TreeOpenPoissonSolver solver(targets, sources, treeparams, solverparams);
        solver.Solve();
        




        /* UNORDERED MAP TEST 
        Kokkos::UnorderedMap<int, double> map;
        map.insert(1, 1.5);
        map.insert(2, 2.5);

        double& ref = map.value_at(map.find(1));
        ref = 12.5;
        
        std::cout << map.value_at(map.find(1)) << " " << map.value_at(map.find(2)) << std::endl;
        */

        /* OCTREE CONSTRUCTION TEST 
        typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
        playout_type PLayout;
        ippl::OrthoTreeParticle particles(PLayout);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif1(0.25, 0.5);
        std::uniform_real_distribution<double> unif2(0, 0.25);

        unsigned int n=1000;
        particles.create(n);
        for(unsigned int i=0; i<n; ++i){
            particles.R(i) = ippl::Vector<double, 3>{unif2(eng),    unif2(eng),    unif2(eng)};
            particles.rho(i) = 0;
        }

        ippl::OrthoTree tree(particles,  2 , 10, 20, ippl::BoundingBox<3>{{0,0,0},{1,1,1}});
        tree.PrintStructure();
        */
        

        
        

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