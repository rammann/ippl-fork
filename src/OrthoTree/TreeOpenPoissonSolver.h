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

        // radius and sigma at the coarsest level
        double r0_m;
        double sig0_m;

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

            r0_m = (max-min)/2.0;
            sig0_m = r0_m/(Kokkos::log(1/eps_m));
            
            eps_m = solverparams.get<double>("eps");

        }
    
    public: // Solve

        void Solve(){
            Farfield();
            FarfieldExplicit();
        }

        void Farfield(){
            
            // Number of Fourier nodes as defined in (3.36)
            int nf = static_cast<int>(Kokkos::ceil(6/Kokkos::numbers::pi * Kokkos::log(1/eps_m)));
            constexpr unsigned int dim = 3;

            
            // Define types for Fourier space field and NUFFT
            using mesh_type                 = ippl::UniformCartesian<double, dim>;
            using centering_type            = mesh_type::DefaultCentering;       
            using vector_type               = ippl::Vector<double, 3>;
            using fourier_field_type        = ippl::Field<Kokkos::complex<double>, dim, mesh_type, centering_type>;
            using nufft_type                = ippl::NUFFT<3,double,mesh_type, centering_type>;

            
            // Index space
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


            // Mesh holds the information about the geometry of the field
            // The mesh is [-nf/2, ..., 0, ..., nf/2]^3
            vector_type hx = {1.0, 1.0, 1.0};
            vector_type origin = {
                -static_cast<double>(nf/2), 
                -static_cast<double>(nf/2), 
                -static_cast<double>(nf/2)
            };
            mesh_type mesh(owned, hx, origin);


            // Define field for Fourier transform of sources g(k) = F{rho(x)}
            // and Fourier transform of the far-field u(k) = {w * g} (k)
            fourier_field_type field_g(mesh, layout);
            fourier_field_type field_u(mesh, layout);
            field_u = 0;
            auto gview = field_g.getView();
            auto uview = field_u.getView();


            // C_tilde = C + b*sig is a constant used in w(k) : (3.20) & Lemma 3.4
            const double b = 6;
            const double Ct = 3 * Kokkos::pow(2 * r0_m,2) + b * sig0_m;

            
            // Setup Type 1 Nufft for Fourier transform of the sources g(k) = F{rho(x)}(k)
            ippl::ParameterList fftParams;
            fftParams.add("use_finufft_defaults", true); 
            int type1 = 1; // NUFFT type 1
            std::unique_ptr<nufft_type> nufft1 = std::make_unique<nufft_type>(layout, sources_m.getTotalNum(), type1, fftParams);

            
            // Type 1 NUFFT
            nufft1->transform(sources_m.R, sources_m.rho, field_g);

            
            // Component-wise multiplication of the Fourier-space fields g(k) and w(k)
            // The result is the Fourier transform of the far field F{u} (k) = {w * g} (k)
            Kokkos::parallel_for("Calculate u_hat = g_hat * w", field_g.getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k){
                    
                    // Convert index (i,j,k) to frequency (k_x,k_y,k_z)
                    const double kx = -static_cast<double>(nf/2) + i;
                    const double ky = -static_cast<double>(nf/2) + j;
                    const double kz = -static_cast<double>(nf/2) + k;

                    // Calculation of w(k)
                    double kabs = Kokkos::sqrt(kx * kx + ky * ky + kz * kz);
                    double w = 1.0/(Kokkos::pow(Kokkos::numbers::pi,2)) * 
                        Kokkos::pow( (Kokkos::sin(Ct * kabs * 0.5) / kabs) , 2) * 
                        Kokkos::exp(-0.25 * Kokkos::pow(kabs * sig0_m,2));

                    // Component wise multiplication
                    uview(i,j,k) = w * gview(i,j,k);
                    //std::cout << uview(i,j,k) << "\n";
                });
            
            
            // Setup Type 2 NUFFT for transform back onto the target points in real space
            int type2 = 2;
            std::unique_ptr<nufft_type> nufft2 = std::make_unique<nufft_type>(layout, targets_m.getTotalNum(), type2, fftParams);

            // Type 2 NUFFT
            nufft2->transform(targets_m.R, targets_m.rho, field_u);
        }

        void FarfieldExplicit(){
            Kokkos::View<double*> targetValues("Explicit farfield solution", targets_m.getTotalNum());
            Kokkos::parallel_for("Compute explicit farfield", targets_m.getTotalNum(), 
                KOKKOS_LAMBDA(const int t){
                    for(int s=0; s<sources_m.getTotalNum(); ++s){
                        double r = Kokkos::sqrt(Kokkos::pow(targets_m.R(t)(0)-sources_m.R(s)(0),2) +
                                                Kokkos::pow(targets_m.R(t)(1)-sources_m.R(s)(1),2) +
                                                Kokkos::pow(targets_m.R(t)(2)-sources_m.R(s)(2),2));
                        targetValues(t) += sources_m.rho(s) * std::erf(r/sig0_m)/r;
                    }
                });
            
            for(int t=0; t<targets_m.getTotalNum(); ++t){
                std::cout << targets_m.rho(t) << " " << targetValues(t) << "\n";
            }
        }

         



        
    };
    
    
    
} // namespace ippl















#endif