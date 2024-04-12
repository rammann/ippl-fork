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

        unsigned int dim_m = 3;



    public: // Constructors

        /**
         * 1. Init Octree
         * 2. Setup Solve
        */
        TreeOpenPoissonSolver(particle_type particles, unsigned int tidx, ParameterList treeparams, ParameterList solverparams) : 
        particles_m(particles)
        {

            // Init tree
            auto min = treeparams.get<double>("boxmin");
            auto max = treeparams.get<double>("boxmax");
            OrthoTree tree_m(particles, treeparams.get<int>("maxdepth"), treeparams.get<int>("maxleafelements"), BoundingBox<3>{{min,min,min},{max,max,max}});
            tree_m.PrintStructure();

            //dim_m = tree_m.GetDim();

            ntargets_m = particles.getLocalNum() - tidx;
            nsources_m = tidx;
            
            eps_m = solverparams.get<double>("eps");

            std::cout << nsources_m << "\n";

        }
    
    public: // Solve

        void Solve(){
            Farfield();
        }

        void Farfield(){
            
            std::cout << "Farfield \n";
            int nf = Kokkos::ceil(6/Kokkos::numbers::pi * Kokkos::log(1/eps_m));
            int Nf = Kokkos::pow(nf, dim_m);
            constexpr unsigned int dim = 3;

            // =============== Step 1: Transform sources into Fourier space =============== //
            
            // Define mesh and centering types for the field type
            using Mesh_t               = ippl::UniformCartesian<double, dim>;
            using Centering_t          = Mesh_t::DefaultCentering;
            using Vector_t             = ippl::Vector<double, 3>;
            
            // Grid Points in Fourier space
            ippl::Vector<int, dim> pt = {nf, nf, nf};
            ippl::Index I(pt[0]);
            ippl::Index J(pt[1]);
            ippl::Index K(pt[2]);
            ippl::NDIndex<dim> owned(I, J, K);
            
            // Specify parallel dimensions (for MPI?)
            std::array<bool, dim> isParallel;  
            isParallel.fill(false);
            
            // Field Layout
            ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

            // Grid spacing of 1 => Fourier space grid [-nf, ..., 0, ..., nf]^3
            std::array<double, dim> dx = {1, 1, 1}; 
            Vector_t hx = {dx[0], dx[1], dx[2]};
            Vector_t origin = {0, 0, 0};
            ippl::UniformCartesian<double, 3> mesh(owned, hx, origin);
            typedef ippl::Field<Kokkos::complex<double>, dim, Mesh_t, Centering_t> field_type;

            
        }

         



        
    };
    
    
    
} // namespace ippl















#endif