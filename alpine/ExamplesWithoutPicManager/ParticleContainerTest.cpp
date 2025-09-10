// Particle Container Test
// Usage:
//          srun ./ParticleContainerTest
//              <nsp> <ppsp> <Nt> 
//     
//          nsp   = No. particle species
//          ppsp  = No. particles per species
//          Nt    = No. timesteps



#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim = 3;

const char* TestName = "ParticleContainer";

// typedefs
template <typename T, unsigned Dim = 3>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

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
        setSignalHandler();

        Inform msg("ParticleContainerTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();
        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);
        
        unsigned int nSp = std::atoi(argv[arg++]); 
        size_type ppSp = std::atoll(argv[arg++]);
    }
    ippl::finalize();

    return 0;
}
