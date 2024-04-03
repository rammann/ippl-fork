/**
 * Class for the OrthoTree - based solver.
*/

#ifndef TREEOPENPOISSONSOLVER
#define TREEOPENPOISSONSOLVER
#include "Utility/ParameterList.h"

namespace ippl
{
    template<typename point_type>
    class TreeOpenPoissonSolver
    {
    private:
        
        unsigned int tidx_m; // idx of first target point


    public: // Constructors

        /**
         * 1. Init Octree
         * 2. Setup Solve
        */
        TreeOpenPoissonSolver(point_type points, unsigned int tidx, ParameterList treeparams): tidx_m(tidx){

            auto min = treeparams.get<double>("boxmin");
            auto max = treeparams.get<double>("boxmax");
            OrthoTree tree(points, treeparams.get<int>("maxdepth"), treeparams.get<int>("maxleafelements"), BoundingBox<3>{{min,min,min},{max,max,max}});
            tree.PrintStructure();

        }
    
    public: // Solve


        
    };
    
    
    
} // namespace ippl















#endif