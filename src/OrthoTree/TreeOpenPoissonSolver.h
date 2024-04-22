/**
 * Class for the OrthoTree - based solver.
*/

#ifndef TREEOPENPOISSONSOLVER
#define TREEOPENPOISSONSOLVER
#include "Utility/ParameterList.h"
#include <unordered_map>

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
            tree_m.PrintStructure();

            // Precision
            eps_m = solverparams.get<double>("eps");
            
            // Radius of the box and sigma at the coarsest level
            r0_m = (max-min)/2.0;
            sig0_m = r0_m/Kokkos::sqrt((Kokkos::log(1/eps_m)));
            
        }
    
    public: // Solve

        void Solve(){
            Farfield();
            //FarfieldExplicit();
            DifferenceKernel();
            ResidualKernel();
            //PrintSolution();
            ExplicitSolution();
        }

        void Farfield(){

            std::cout << "Starting farfield calculation" << "\n";
            
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

            std::cout << "Finished farfield calculation" << "\n\n";
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
                std::cout << targetValues(t) << " " << targets_m.rho(t) <<"\n";
            }
        }

        void DifferenceKernel(){

            std::cout << "Starting difference calculation" << "\n";
            const unsigned int dim = 3;    
            int nf = static_cast<int>(Kokkos::ceil(6 / Kokkos::numbers::pi * Kokkos::log(1/eps_m)));

            // Iterate through levels of the tree
            for(unsigned int depth=0; depth <= tree_m.GetMaxDepth(); ++depth){

                std::cout << "At depth " << depth << "\n";

                // Depth dependent variables
                double r = r0_m / Kokkos::pow(2, depth);
                double h = 0.9 * 4 * Kokkos::numbers::pi / (3 * r);
                

                // nodekeys is a vector holding the morton ids of the internal nodes at current depth
                Kokkos::vector<morton_node_id_type> nodekeys = tree_m.GetInternalNodeAtDepth(depth);


                // keytoidx maps morton ids to their index in nodekeys
                std::unordered_map<morton_node_id_type, unsigned int> keytoidx;
                for(unsigned int i=0; i<nodekeys.size(); ++i){
                    keytoidx[nodekeys[i]] = i;
                }
                    
                // Container for outgoing expansion
                Kokkos::UnorderedMap<morton_node_id_type, fourier_field_type> Phi;
                
                // Outgoing expansion at depth
                for(unsigned int i=0; i<nodekeys.size(); ++i){

                    // Morton key of current internal node
                    morton_node_id_type key = nodekeys[i];
                    std::cout << "Performing outgoing expansion for node " << key << "\n";

                    // Get node center
                    std::cout << "Getting node center" << "\n";
                    ippl::Vector<double,dim> center = tree_m.GetNode(key).GetCenter();

                    // Get souce ids in this node
                    std::cout << "Getting source ids" << "\n";
                    Kokkos::vector<entity_id_type> idSources = tree_m.CollectSourceIds(key);
                
                    // Create source particles with positions relative to node center
                    std::cout << "Creating particles relative to center" << "\n";
                    playout_type PLayout;
                    particle_type relSources(PLayout);
                    relSources.create(idSources.size());
                    for(unsigned int i=0; i<idSources.size(); ++i){
                        relSources.R(i) = h * (sources_m.R(idSources[i]) - center);
                        relSources.rho(i) = sources_m.rho(idSources[i]);
                    }
                    
                    

                    // Create Fourier-space field for outgoing expansion
                    ippl::Vector<int, dim> pt = {nf, nf, nf};
                    ippl::Index I(pt[0]); 
                    ippl::Index J(pt[1]); 
                    ippl::Index K(pt[2]);
                    ippl::NDIndex<3> owned(I, J, K);

                    std::array<bool, dim> isParallel;  
                    isParallel.fill(false);

                    ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

                    vector_type hx = {1.0, 1.0, 1.0};
                    vector_type origin = {
                        -static_cast<double>(nf/2), 
                        -static_cast<double>(nf/2), 
                        -static_cast<double>(nf/2)
                    };
                    mesh_type mesh(owned, hx, origin);

                    fourier_field_type fieldPhi(mesh, layout,0);
                    fieldPhi = 0;


                    // Initialise NUFFT 1
                    ippl::ParameterList fftParams;
                    fftParams.add("use_finufft_defaults", true); 
                    int type1 = 1; 
                    std::unique_ptr<nufft_type> nufft1 = std::make_unique<nufft_type>(layout, sources_m.getTotalNum(), type1, fftParams);


                    // Perform NUFFT 1
                    std::cout << "Performing NUFFT Type 1" << "\n";
                    nufft1->transform(relSources.R, relSources.rho, fieldPhi);
                    

                    // Insert outgoing expansion into map
                    std::cout << "Inserting (key, field) pair into map" << "\n";
                    Phi.insert(key, fieldPhi);
                    
                } // Loop over nodes for outgoing expansion
                

                Kokkos::UnorderedMap<morton_node_id_type, fourier_field_type> Psi;

                // Incoming expansion at depth
                for(unsigned int i=0; i<nodekeys.size(); ++i){

                    // Morton key of current node
                    morton_node_id_type key = nodekeys[i];

                    std::cout << "Performing incoming expansion for key " << key << "\n";

                    // Vector of colleague keys
                    Kokkos::vector<morton_node_id_type> colleaguekeys = tree_m.GetColleagues(key);
                    std::cout << "Number of colleagues is " << colleaguekeys.size() << "\n";

                    // Create Fourier-space field for incoming expansion
                    ippl::Vector<int, dim> pt = {nf, nf, nf};
                    ippl::Index I(pt[0]); 
                    ippl::Index J(pt[1]); 
                    ippl::Index K(pt[2]);
                    ippl::NDIndex<3> owned(I, J, K);

                    std::array<bool, dim> isParallel;  
                    isParallel.fill(false);

                    ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

                    vector_type hx = {1.0, 1.0, 1.0};
                    vector_type origin = {
                        -static_cast<double>(nf/2), 
                        -static_cast<double>(nf/2), 
                        -static_cast<double>(nf/2)
                    };
                    mesh_type mesh(owned, hx, origin);

                    fourier_field_type fieldPsi(mesh, layout,0);
                    fieldPsi = 0;
                    auto PsiView = fieldPsi.getView();

                    // Loop over colleagues to calculate incoming expansion
                    for(unsigned int c=0; c<colleaguekeys.size(); ++c){
                        
                        // Morton key of colleague
                        morton_node_id_type colkey = colleaguekeys[c];
                        std::cout << "Gathering expansion from colleague " << colkey << "\n";

                        // Colleague's outgoing expansion view
                        if(!Phi.exists(colkey)) continue;
                        auto PhiView = Phi.value_at(Phi.find(colkey)).getView();

                        // Difference of centers of node and its colleague
                        ippl::Vector<double,3> delta = tree_m.GetNode(key).GetCenter() - tree_m.GetNode(colkey).GetCenter();

                        // Incoming expansion for current colleague
                        Kokkos::parallel_for("Calculate incoming expansion", fieldPsi.getFieldRangePolicy(),
                        KOKKOS_LAMBDA(const int i, const int j, const int k){
                                
                            // Transform multi-index to k vector
                            const double kx = -static_cast<double>(nf/2) + i;
                            const double ky = -static_cast<double>(nf/2) + j;
                            const double kz = -static_cast<double>(nf/2) + k;

                            // w value
                            double w = Kokkos::pow(Kokkos::numbers::pi*2,-3) * D(depth, kx, ky, kz);

                            // Dot product of (kx,ky,kz) and (centerdifference)
                            double t = kx * delta[0] + ky * delta[1] + kz * delta[2];

                            // i
                            Kokkos::complex<double> I;
                            I.real() = 0; I.imag() = 1;

                            PsiView(i,j,k) += w * Kokkos::exp(I * h * t) * PhiView(i,j,k);

                        }); // Incoming expansion loop

                    } // Loop over colleagues of node

                    std::cout << "Inserting (key,field) pair" << "\n";
                    Psi.insert(key, fieldPsi);
                    
                } // Loop over nodes for incoming expansion

                // Nufft back onto target on each node
                for(unsigned int i=0; i<nodekeys.size(); ++i){

                    // Morton key of current internal node
                    morton_node_id_type key = nodekeys[i];

                    // Get incoming expansion field
                    if(!Psi.exists(key)) continue;
                    auto fieldPsi = Psi.value_at(Psi.find(key));
                    
                    // Get node center
                    ippl::Vector<double,dim> center = tree_m.GetNode(key).GetCenter();

                    // Get souce ids in this node
                    Kokkos::vector<entity_id_type> idTargets = tree_m.CollectTargetIds(key);

                    // Create source particles with positions relative to node center
                    playout_type PLayout;
                    particle_type relTargets(PLayout);
                    relTargets.create(idTargets.size());
                    for(unsigned int i=0; i<idTargets.size(); ++i){
                        relTargets.R(i) = h * (targets_m.R(idTargets[i]) - center);
                        relTargets.rho(i) = 0;
                    }

                    // Create Fourier-space field for outgoing expansion
                    ippl::Vector<int, dim> pt = {nf, nf, nf};
                    ippl::Index I(pt[0]); 
                    ippl::Index J(pt[1]); 
                    ippl::Index K(pt[2]);
                    ippl::NDIndex<3> owned(I, J, K);

                    std::array<bool, dim> isParallel;  
                    isParallel.fill(false);

                    ippl::FieldLayout<dim> layout(MPI_COMM_WORLD, owned, isParallel);

                    vector_type hx = {1.0, 1.0, 1.0};
                    vector_type origin = {
                        -static_cast<double>(nf/2), 
                        -static_cast<double>(nf/2), 
                        -static_cast<double>(nf/2)
                    };
                    mesh_type mesh(owned, hx, origin);

                    ippl::ParameterList fftParams;
                    fftParams.add("use_finufft_defaults", true); 
                    int type2 = 2; 
                    std::unique_ptr<nufft_type> nufft2 = std::make_unique<nufft_type>(layout, sources_m.getTotalNum(), type2, fftParams);

                    // std::cout << "Total number of target points is " <<relTargets.getTotalNum() << "\n";
                    // Perform NUFFT 1
                    nufft2->transform(relTargets.R, relTargets.rho, fieldPsi);

                    

                    // Add this contribution to target values
                    Kokkos::parallel_for("Add contribution to target values", idTargets.size(),
                    KOKKOS_LAMBDA(unsigned int t){
                        targets_m.rho(idTargets[t]) += relTargets.rho(t);
                    });
                    std::cout << "HERE IS THE ERROR" << "\n";
                }


                
            } // Loop over depth

        }

        
        void ResidualKernel(){

            Kokkos::vector<morton_node_id_type> leafnodes = tree_m.GetLeafNodes();

            //Kokkos::parallel_for("Loop over leaf nodes for residual contribution", leafnodes.size(),
            //KOKKOS_LAMBDA(unsigned int i){
            for(unsigned int i=0; i<leafnodes.size(); ++i){

                // Leaf key
                morton_node_id_type leafkey = leafnodes[i];

                // Leaf node
                OrthoTreeNode leafnode = tree_m.GetNode(leafkey);

                // Depth
                depth_type depth = tree_m.GetDepth(leafkey);
               
                // Find colleagues and coarse neighbours
                Kokkos::vector<morton_node_id_type> coarseneighbours = tree_m.GetCoarseNbrs(leafkey);
                Kokkos::vector<morton_node_id_type> colleagues = tree_m.GetColleagues(leafkey);

                // Collect source ids
                Kokkos::vector<entity_id_type> sourceids = tree_m.CollectSourceIds(leafkey);
                Kokkos::vector<entity_id_type> targetids = {};
                targetids.reserve(static_cast<unsigned int>((coarseneighbours.size() + colleagues.size()) * 0.4 * tree_m.GetMaxElementsPerNode()));
                

                // Incremement targets within coarse neighbours
                for(unsigned int nidx=0; nidx<coarseneighbours.size(); ++nidx){
                    
                    // Coarse leaf neighbour key
                    morton_node_id_type key = coarseneighbours[nidx];

                    // Collect Targets in this coarse leaf neighbour
                    Kokkos::vector<entity_id_type> tids = tree_m.CollectTargetIds(key);

                    // Fill target ids into vector
                    for(unsigned int tidx=0; tidx<tids.size(); ++tidx){
                        targetids.push_back(tids[tidx]);
                    }
                }

                // Increment targets within colleagues
                for(unsigned int nidx=0; nidx<colleagues.size(); ++nidx){
                    
                    // Coarse leaf neighbour key
                    morton_node_id_type key = colleagues[nidx];

                    // Collect Targets in this coarse leaf neighbour
                    Kokkos::vector<entity_id_type> tids = tree_m.CollectTargetIds(key);

                    // Fill target ids into vector
                    for(unsigned int tidx=0; tidx<tids.size(); ++tidx){
                        targetids.push_back(tids[tidx]);
                    }
                }

                // Calculate Residual Contribution
                for(unsigned int sidx=0; sidx<sourceids.size(); ++sidx){

                    entity_id_type sourceid = sourceids[sidx];
                    vector_type sourceR = sources_m.R(sourceid);
                    double sourceRho = sources_m.rho(sourceid);

                    //std::cout << "Current Source " << sourceid << " with charge " << sourceRho << "\n";

                    for(unsigned int tidx=0; tidx<targetids.size(); ++tidx){

                        entity_id_type targetid = targetids[tidx];
                        vector_type targetR = targets_m.R(targetid);
                        ippl::Vector<double, 3> deltaR = targetR - sourceR;
                        double r = Kokkos::sqrt(deltaR[0] * deltaR[0] + deltaR[1] * deltaR[1] + deltaR[2] * deltaR[2]);

                        targets_m.rho(targetid) += R(depth, r) * sourceRho;
                    }
                }
            }
        }

        void ExplicitSolution(){
            Kokkos::View<double*> targetValues("Explicit farfield solution", targets_m.getTotalNum());

            Kokkos::parallel_for("Compute explicit solution", targets_m.getTotalNum(), 
                KOKKOS_LAMBDA(const unsigned int t){
                    double temp = 0;
                    for(unsigned int s=0; s<sources_m.getTotalNum(); ++s){
                        double r = Kokkos::sqrt(Kokkos::pow(targets_m.R(t)(0)-sources_m.R(s)(0),2) +
                                                Kokkos::pow(targets_m.R(t)(1)-sources_m.R(s)(1),2) +
                                                Kokkos::pow(targets_m.R(t)(2)-sources_m.R(s)(2),2));
                        temp += sources_m.rho(s) * 1/r;
                    }
                    targetValues(t) = temp;
                });
            
            for(unsigned int t=0; t<targets_m.getTotalNum(); ++t){
                std::cout <<  targetValues(t) << " " << targets_m.rho(t) <<"\n";
            }
        }

    private: // Aid functions

        inline double D(unsigned int l, double kx, double ky, double kz){
            double sigl = sig0_m * Kokkos::pow(0.5,l);
            double sigl1 = sigl * 0.5;
            if( Kokkos::abs(kx) < std::numeric_limits<double>::epsilon() && 
                Kokkos::abs(ky) < std::numeric_limits<double>::epsilon() &&
                Kokkos::abs(kz) < std::numeric_limits<double>::epsilon())
            {
                return Kokkos::numbers::pi * (sigl - sigl1);
            }
            
            else
            {
                double k2 = kx * kx + ky * ky + kz * kz;
                return 4 * Kokkos::numbers::pi / k2  * (Kokkos::exp(-k2 * sigl1 * sigl1 * 0.25) - Kokkos::exp(-k2 * sigl * sigl * 0.25));
            }
        }

        inline double R(unsigned int l, double r){
            double sigl = sig0_m * Kokkos::pow(0.5,l);
            return std::erfc(r / sigl)/r;
        }

        void PrintSolution(){
            for(unsigned int i=0; i<targets_m.getTotalNum(); ++i){
                std::cout << targets_m.rho(i) << "\n";
            }
        }




    };
    
    
    
} // namespace ippl















#endif