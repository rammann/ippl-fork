/**
 * Class for the OrthoTree - based solver.
*/

#ifndef TREEOPENPOISSONSOLVER
#define TREEOPENPOISSONSOLVER
#include "Utility/ParameterList.h"

namespace ippl
{
    class TreeOpenPoissonSolver
    {

    public: // Types

        // Define types for Fourier space field and NUFFT
        using mesh_type                 = ippl::UniformCartesian<double, 3>;
        using centering_type            = mesh_type::DefaultCentering;       
        using vector_type               = ippl::Vector<double, 3>;
        using fourier_field_type        = ippl::Field<Kokkos::complex<double>, 3, mesh_type, centering_type>;
        using nufft_type                = ippl::NUFFT<3,double,mesh_type, centering_type>;
        using particle_type             = OrthoTreeParticle<>;
        using playout_type              = ippl::ParticleSpatialLayout<double, 3>;

    private:

        // Octree 
        OrthoTree tree_m;
        
        // Particles
        particle_type sources_m;
        particle_type targets_m;

        // Views of different contributions
        Kokkos::View<double*> farfield_m;
        Kokkos::View<double*> difference_m;
        Kokkos::View<double*> residual_m;


        // Precision
        double eps_m;


        // Radius of the box and sigma at the coarsest level
        double r0_m;
        double sig0_m;

    public: // Constructors

        TreeOpenPoissonSolver(particle_type targets, particle_type sources, ParameterList treeparams, ParameterList solverparams)
        {
            // Target and source particles
            targets_m = targets;
            sources_m = sources;

            // Views
            farfield_m = Kokkos::View<double*>("Farfield contribution", targets.getTotalNum());
            difference_m = Kokkos::View<double*>("Difference contribution", targets.getTotalNum());
            residual_m = Kokkos::View<double*>("Difference contribution", targets.getTotalNum());
            


            // For the octree a view with all particle positions needs to be created
            // allParticles = [target(0), target(1), ..., target(#targets-1), sources(0), ..., sources(#sources-1)]
            // indices      = [0,        1,        , ..., #targets-1,       , #targets ,  ..., #targets + #sources -1 ]
            playout_type PLayout;
            particle_type allParticles(PLayout);
            allParticles.create(targets.getTotalNum() + sources.getTotalNum());
            Kokkos::parallel_for("Fill sources and targets into one view for octree construction",
                allParticles.getTotalNum(), [=](unsigned int i){
                    if (i < targets.getTotalNum()) {
                        allParticles.R(i) = targets.R(i); 
                        allParticles.rho(i) = 0.0; // charge is not used for the octree
                    }
                    else {
                        allParticles.R(i) = sources.R(i-targets.getTotalNum());
                        allParticles.rho(i) = 0.0;
                    }
                });

            // Tree Construction
            auto min = treeparams.get<double>("boxmin");
            auto max = treeparams.get<double>("boxmax");
            auto maxdepth = treeparams.get<int>("maxdepth");
            auto maxleafele = treeparams.get<int>("maxleafelements");
            auto sourceidx = treeparams.get<unsigned int>("sourceidx");
            tree_m = OrthoTree(allParticles, sourceidx, maxdepth, maxleafele, BoundingBox<3>{{min,min,min},{max,max,max}});
            //tree_m.PrintStructure();

            // Precision
            eps_m = solverparams.get<double>("eps");
            
            // Radius of the box and sigma at the coarsest level
            r0_m = (max-min)/2.0;
            sig0_m = r0_m/Kokkos::sqrt((Kokkos::log(1/eps_m)));
            
        }
    
    public: // Solve

        void Solve(){
            //Farfield();
            //FarfieldExplicit();
            DifferenceKernel();
        }

        void Farfield(){
            
            // Number of Fourier nodes as defined in (3.36)
            int nf = static_cast<int>(Kokkos::ceil(4 * Kokkos::log(1/eps_m))) * 2;
            constexpr unsigned int dim = 3;


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
            fourier_field_type field_g(mesh, layout,0);
            fourier_field_type field_u(mesh, layout,0);
            field_u = 0;
            auto gview = field_g.getView();
            auto uview = field_u.getView();


            // C_tilde = C + b*sig is a constant used in w(k) : (3.20) & Lemma 3.4
            const double b = 6;
            const double Ct = Kokkos::sqrt(3 * 2 * r0_m) + b * sig0_m;

            
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

                    //std::cout << kx << " " << ky << " " << kz << "\n";

                    // Calculation of w(k)
                    double kabs = Kokkos::sqrt(kx * kx + ky * ky + kz * kz) + std::numeric_limits<double>::epsilon();
                    double w =  Kokkos::pow(Kokkos::numbers::pi, -2) * 
                                Kokkos::pow( (Kokkos::sin(Ct * kabs * 0.5) / kabs) , 2) * 
                                Kokkos::exp(-0.25 * Kokkos::pow(kabs * sig0_m,2));

                    // Component wise multiplication
                    uview(i,j,k) = w * gview(i,j,k);
                    //std::cout << uview(i,j,k) << " " << w << gview(i,j,k) << "\n";
                });
            
            
            // Setup Type 2 NUFFT for transform back onto the target points in real space
            int type2 = 2;
            std::unique_ptr<nufft_type> nufft2 = std::make_unique<nufft_type>(layout, targets_m.getTotalNum(), type2, fftParams);

            // Type 2 NUFFT
            nufft2->transform(targets_m.R, targets_m.rho, field_u);

            // Copy result to farfield view
            Kokkos::parallel_for("Copy farfield data to view", targets_m.getTotalNum(),
            KOKKOS_LAMBDA(unsigned int i){
                farfield_m(i) = targets_m.rho(i);
                targets_m.rho(i) = 0.0;
            });
        }

        void FarfieldExplicit(){
            Kokkos::View<double*> targetValues("Explicit farfield solution", targets_m.getTotalNum());

            Kokkos::parallel_for("Compute explicit farfield", targets_m.getTotalNum(), 
                KOKKOS_LAMBDA(const unsigned int t){
                    double temp = 0;
                    for(unsigned int s=0; s<sources_m.getTotalNum(); ++s){
                        double r = Kokkos::sqrt(Kokkos::pow(targets_m.R(t)(0)-sources_m.R(s)(0),2) +
                                                Kokkos::pow(targets_m.R(t)(1)-sources_m.R(s)(1),2) +
                                                Kokkos::pow(targets_m.R(t)(2)-sources_m.R(s)(2),2));
                        temp += sources_m.rho(s) * std::erf(r/sig0_m)/r;
                    }
                    targetValues(t) = temp;
                });
            
            for(unsigned int t=0; t<targets_m.getTotalNum(); ++t){
                std::cout << farfield_m(t) << " " << targetValues(t) << " " << targets_m.rho(t) <<"\n";
            }
        }

        void DifferenceKernel(){

            const unsigned int dim = 3;    
            int nf = static_cast<int>(Kokkos::ceil(6 / Kokkos::numbers::pi * Kokkos::log(1/eps_m)));
            
            for(unsigned int depth=1; depth <= tree_m.GetMaxDepth(); ++depth){

                double r = r0_m / Kokkos::pow(2, depth);
                double h = 0.9 * 4 * Kokkos::numbers::pi / (3 * r);
                
                Kokkos::vector<morton_node_id_type> nodekeys = tree_m.GetInternalNodeAtDepth(depth);
                

                Kokkos::parallel_for("Loop over nodes of outoging expansion", nodekeys.size(),
                    KOKKOS_LAMBDA(unsigned int i){

                        // Get node center 
                        OrthoTreeNode node = tree_m.GetNode(nodekeys[i]);
                        ippl::Vector<double,3> center = {
                            node.boundingbox_m.Min[0] + (node.boundingbox_m.Max[0]-node.boundingbox_m.Min[0]) * 0.5,
                            node.boundingbox_m.Min[1] + (node.boundingbox_m.Max[1]-node.boundingbox_m.Min[1]) * 0.5,
                            node.boundingbox_m.Min[2] + (node.boundingbox_m.Max[2]-node.boundingbox_m.Min[2]) * 0.5
                        };

                        // Collect source ids
                        Kokkos::vector<entity_id_type> sourceids = tree_m.CollectSourceIds(nodekeys[i]);
                        
                        // Create new source points relative to box center for NUFFT
                        playout_type PLayout;
                        particle_type relativesources(PLayout);
                        relativesources.create(sourceids.size());
                        for(unsigned int i=0; i<sourceids.size(); ++i){
                            relativesources.R(i) = h * (sources_m.R(sourceids[i]) - center);
                            relativesources.rho(i) = sources_m.rho(sourceids[i]);
                        }

                        // Create Fourier-space field for outgoing expansion
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


                        // Fourier-space field for outgoing expansion of this node
                        fourier_field_type field_phi(mesh, layout,0);
                        field_phi = 0;
                        auto phiview = field_phi.getView();

                        // Perform NUFFT1
                        
                    });


                
                
            }
                // At each level, get all non-leaf nodes
                    // For each node compute the outgoing expansion

                    // For each compute the incoming expansion
        }

        void ResidualKernel(){
            Kokkos::vector<morton_node_id_type> leafnodes = tree_m.GetLeafNodes();

            Kokkos::parallel_for("Loop over leaf nodes for residual contribution", leafnodes.size(),
                KOKKOS_LAMBDA(unsigned int i){
                    OrthoTreeNode leaf = tree_m.GetNode(leafnodes[i]);
                    // Find colleagues and coarse neighbours

                    // Find all sources withing those neighbours

                    // Calculate Residual Contribution
                });
        }



        
    };
    
    
    
} // namespace ippl















#endif