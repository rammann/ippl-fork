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
        using playout_type = ippl::ParticleSpatialLayout<double, 3>;

    private:

        OrthoTree tree_m;
        
        //particle_type particles_m;
        particle_type sources_m;
        particle_type targets_m;

        double eps_m;

        //unsigned int dim_m = 3;

    public: // Constructors

        /**
         * 1. Init Octree
         * 2. Setup Solve
        */
        TreeOpenPoissonSolver(particle_type targets, particle_type sources, ParameterList treeparams, ParameterList solverparams) 
        {
            targets_m = targets;
            sources_m = sources;

            playout_type PLayout;


            // Make one view of all points for the octree construction
            unsigned int idxSource = targets.getTotalNum(); // Index of first source point in allParticles
            particle_type allParticles(PLayout);
            allParticles.create(targets.getTotalNum() + sources.getTotalNum());
            Kokkos::parallel_for("Fill sources and targets into one view for octree construction",
                allParticles.getTotalNum(), [=](unsigned int i){
                    if (i < idxSource) {
                        allParticles.R(i) = targets.R(i); 
                        allParticles.rho(i) = 0.0; // charge is not used for the octree
                    }
                    else {
                        allParticles.R(i) = sources.R(i-idxSource);
                        allParticles.rho(i) = 0.0;
                    }
                });

            // Init tree
            auto min = treeparams.get<double>("boxmin");
            auto max = treeparams.get<double>("boxmax");
            OrthoTree tree_m(allParticles, treeparams.get<int>("maxdepth"), treeparams.get<int>("maxleafelements"), BoundingBox<3>{{min,min,min},{max,max,max}});
            tree_m.PrintStructure();

            //dim_m = tree_m.GetDim();

            
            eps_m = solverparams.get<double>("eps");

        }
    
    public: // Solve

        void Solve(){
            Farfield();
        }

        void Farfield(){
            
            std::cout << "Farfield \n";
            int nf = static_cast<int>(Kokkos::ceil(6/Kokkos::numbers::pi * Kokkos::log(1/eps_m)));
            //int Nf = static_cast<int>(Kokkos::pow(nf, dim_m));
            constexpr unsigned int dim = 3;

            // =============== Step 1: Transform sources into Fourier space =============== //
            
            // Define mesh and centering types for the field type
            using mesh_type                 = ippl::UniformCartesian<double, dim>;
            using centering_type            = mesh_type::DefaultCentering;       
            using vector_type               = ippl::Vector<double, 3>;
            using field_type                = ippl::Field<Kokkos::complex<double>, dim, mesh_type, centering_type>;
            using nufft_type                = ippl::NUFFT<3,double,mesh_type, centering_type>;
            
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


            // Grid spacing of 1 => Fourier space grid [-nf/2, ..., 0, ..., nf/2]^3
            vector_type hx = {1.0, 1.0, 1.0};
            vector_type origin = {
                -static_cast<double>(nf/2), 
                -static_cast<double>(nf/2), 
                -static_cast<double>(nf/2)
            };

            // Fourier-space complex field
            mesh_type mesh(owned, hx, origin);
            field_type field(mesh, layout);

            // Use default parameters
            ippl::ParameterList fftParams;
            fftParams.add("use_finufft_defaults", true); 
            int type = 1; // NUFFT type 1
            
            std::unique_ptr<nufft_type> nufft = std::make_unique<nufft_type>(layout, sources_m.getTotalNum(), type, fftParams);
            nufft->transform(sources_m.R, sources_m.rho, field);
            
        }

         



        
    };
    
    
    
} // namespace ippl















#endif