
// Particle Communication Test
//   Usage:
//     srun ./ParticleCommunicationTest
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     lbfreq   = Load balancing frequency i.e., Number of time steps after which particle
//                load balancing should happen
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./ParticleCommunicationTest 128 128 128 10000 100 20 
//

#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim=3;

const char* TestName = "ParticleCommunicationTest";

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type vals;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T start, end;

    // Initialize all members
    generate_random(view_type vals_, GeneratorPool rand_pool_, T start_, T end_)
        : vals(vals_)
        , rand_pool(rand_pool_)
        , start(start_)
        , end(end_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        // Draw samples numbers from the pool as double in the range [start, end)
        for (unsigned d = 0; d < Dim; ++d) {
            vals(i)[d] = rand_gen.drand(start[d], end[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};


int main(int argc, char* argv[]) {
    
    ippl::initialize(argc, argv);
    {
           
        setSignalHandler();
        
        Inform msg("ParticleCommunicationTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();
        int arg = 1;

        // 1. Setup ============================================================================ // 
        
        // 1.1 Domain and Layout Setup ========================================================= //
        
        // Cells per Dimension 
        Vector_t<int, Dim> nr; 
        for(unsigned d = 0; d < Dim; d++){
            nr[d] = std::atoi(argv[arg++]);
        }
        
        // ND Index  
        ippl::NDIndex<Dim> domain;
        for(unsigned i=0; i<Dim; i++){
            domain[i] = ippl::Index(nr[i]);
        } 
        
        // Domain Bounds 
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(1.0);

        // Cell Sizes
        Vector_t<double, Dim> hr = rmax / nr;
        Vector_t<double, Dim> origin = rmin;
        const double dt = 1.0;
        
        // Parallel Dimensions 
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        // Periodic Dimensions 
        const bool isAllPeriodic = true;
        
        // Mesh, Field and Particle Layouts 
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);
      
        // Calculate Rank Bounds 
        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        Vector_t<double, Dim> Rmin, Rmax;
        for (unsigned d = 0; d < Dim; ++d) {
            Rmin[d] = origin[d] + lDom[d].first() * hr[d];
            Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
        }

        // 1.2 Particle Setup ================================================================== // 
        
        // Total Particles 
        const size_type totalP = std::atoll(argv[arg++]);
        
        // Total Timesteps
        const unsigned int nt = std::atoi(argv[arg++]);

        // Particle Pointer 
        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;
        std::unique_ptr<bunch_type> P; 
        
        // Total Charge (dummy variable, not used)
        double Q           = -1562.5;
        
        // Solver (dummy sovler, not used)
        std::string solver = "CG";

        // Initalize Pointer to Particles 
        P = std::make_unique<bunch_type>(PL, hr, rmin, rmax, isParallel, Q, solver);
       
        // Calculate the nloc particles per rank to be created 
        P->nr_m        = nr;
        size_type nloc = totalP / ippl::Comm->size();
        int rest = (int)(totalP - nloc * ippl::Comm->size());
        if (ippl::Comm->rank() < rest)
            ++nloc;
        
        // create nloc particles per rank
        P->create(nloc);

        // each rank samples its particles positions
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), rand_pool64, Rmin, Rmax));
        Kokkos::fence();
        P->q = P->Q_m / totalP;
        P->P = 0.0;

        P->initializeFields(mesh, FL);

        P->update();

        P->loadbalancefreq_m = std::atoi(argv[arg++]);
        P->initializeORB(FL, mesh);
        bool fromAnalyticDensity = false;
        
        msg << "Particle Communication Test" << endl << "nt= " << nt 
            << " Np= " << totalP << " grid= " << nr 
            << " Loadbalancefreq=" << P->loadbalancefreq_m << endl;
        
        msg << "particles created and initial conditions assigned " << endl;
        
        // 2. Iterations ======================================================================= // 
        
        msg << "Starting iterations ..." << endl;
        for(unsigned int it=0; it<nt; it++){
           
            // Sample Displacement 
            Kokkos::parallel_for(
                P->getLocalNum(),
                generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                    P->P.getView(), rand_pool64, Rmin-Rmax,Rmax-Rmin));
            Kokkos::fence();
           
            // Displace
            //P->R = P->R + P->P;
            
            // Resample Position over entire domain since the boundary conditions are not implemented correctly
            P->R = P->P; 

            // Particle Update and Inter Rank Communication 
            P->update();

            //if (P->balance(totalP, it + 1)) {
            //    msg << "Starting Repartition" << endl;
            //    P->repartition(FL, mesh, fromAnalyticDensity);
            //}
 
            P->scatterCIC(totalP, it + 1, hr);
            P->gatherCIC();
            P->time_m += dt;
        }
       
        msg << "Particle Communication Test: End." << endl;
       
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();
        
    }

    ippl::finalize();
    return 0;

}

