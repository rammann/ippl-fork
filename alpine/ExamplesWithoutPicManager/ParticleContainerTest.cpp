// Particle Container Test
// Usage:
//          srun ./ParticleContainerTest
//              <nt> <nsp> <ppsp> 
//
//          nt    = No. timesteps 
//          nsp   = No. particle species
//          ppsp  = No. particles per species

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"
#include "Utility/TypeUtils.h"

constexpr unsigned Dim = 3;

const char* TestName = "ParticleContainer";

// typedefs
template <unsigned Dim = 3>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim = 3>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim = 3>
using FieldLayout_t = ippl::FieldLayout<Dim>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector_t = ippl::Vector<T, Dim>;

// random generator
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

// Particle Containers
template <class PLayout, typename T, unsigned Dim = 3>
class SecondaryParticleContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    // Base::particle_position_type R = Positions (already defined in Base)
    typename Base::particle_position_type P; // Momenta
    ParticleAttrib<int> Sp; // Particle Species 

    SecondaryParticleContainer(PLayout& pl)
        : Base(pl)
    {
        this->addAttribute(P);
        this->addAttribute(Sp);
    }

    ~SecondaryParticleContainer() {}


};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Message objects
        Inform msg("ParticleContainerTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        // Timers
        auto start = std::chrono::high_resolution_clock::now();
        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);
       
        // Read inputs
        //const double dt              = 1.0;
        int arg=1;
        const unsigned int nt   = std::atoi(argv[arg++]);
        unsigned int nSp        = std::atoi(argv[arg++]); 
        size_type ppSp          = std::atoll(argv[arg++]);
        msg << "Particle Container Test" << endl
            << "nt " << nt << " nSp= " << nSp << " ppSp = " << ppSp << endl;
        
        // Define container pointer
        using container_type = SecondaryParticleContainer<PLayout_t<double,Dim>, double, Dim>;
        std::unique_ptr<container_type> PC;

        // Necessary objects for ParticleSpatialLayout
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++)
            nr[d] = 32;
        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(20.0);
        Vector_t<double, Dim> hr     = rmax / nr;
        Vector_t<double, Dim> origin = rmin;
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);
        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        // Construct particle container
        PC = std::make_unique<container_type>(PL);

        // Create particles
        PC->create(nSp*ppSp);

        // End Timings
        msg << "Particle Container Test: End. " << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
    }
    ippl::finalize();

    return 0;
}
