/**
 * Class for the OrthoTree - based solver.
*/

#ifndef TREEOPENPOISSONSOLVER
#define TREEOPENPOISSONSOLVER
#include "Utility/ParameterList.h"

namespace ippl
{
    //template<typename particle_type=OrthoTreeParticle<>>
    class TreeOpenPoissonSolver
    {

    public: // types

        using particle_type = OrthoTreeParticle<>;

    private:

        OrthoTree tree_m;
        
        particle_type particles_m;

        //unsigned int tidx_m; // idx of first target point
        unsigned int ntargets_m;
        unsigned int nsources_m; // index of first target

        double eps_m;

        unsigned int dim_m;


    public: // Constructors

        /**
         * 1. Init Octree
         * 2. Setup Solve
        */
        TreeOpenPoissonSolver(particle_type particles, unsigned int tidx, ParameterList treeparams, ParameterList solverparams) : 
        particles_m(particles), nsources_m(tidx)
        {

            // Init tree
            auto min = treeparams.get<double>("boxmin");
            auto max = treeparams.get<double>("boxmax");
            OrthoTree tree_m(particles, treeparams.get<int>("maxdepth"), treeparams.get<int>("maxleafelements"), BoundingBox<3>{{min,min,min},{max,max,max}});
            tree_m.PrintStructure();

            dim_m = tree_m.GetDim();

            ntargets_m = particles.getLocalNum() - tidx;
            
            eps_m = solverparams.get<double>("eps");

        

        }
    
    public: // Solve

        void Solve(){
            Farfield();
        }

        void Farfield(){
        
            std::cout << "Farfield \n";
            
            //unsigned int nf = Kokkos::ceil(6/Kokkos::numbers::pi * Kokkos::log(1/eps_m));
            //unsigned int Nf = Kokkos::pow(nf, dim_m);
        }

         



        
    };
    
    
    
} // namespace ippl















#endif