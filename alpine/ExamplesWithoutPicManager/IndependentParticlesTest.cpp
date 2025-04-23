// IndependentParticlesTest
// //   Usage:
//     srun ./IndependentParticlesTest

#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"
// #include "Utility/TypeUtils.h"

constexpr unsigned Dim = 3;

const char* TestName = "IndependentParticleTest";

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

template <class PLayout, typename T, unsigned Dim = 3>
class IndependentParticles : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    // Domain
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    // Charge Attribute
    ParticleAttrib<double> q;
    // Mass Attribute
    ParticleAttrib<double> m;
    // Time Attribute
    ParticleAttrib<double> t;
    // Velocity Attribute
    ParticleAttrib<double> v;

    // Constructor
    IndependentParticles(PLayout& pl, Vector_t<double, Dim> rmin, Vector_t<double, Dim> rmax)
        : Base(pl)
        , rmin_m(rmin)
        , rmax_m(rmax)
        , newParticles_m(0) {
        // Register Attributes
        this->addAttribute(q);
        this->addAttribute(m);
        this->addAttribute(t);
        this->addAttribute(v);
    }

    void spawnParticles(size_type nLocal) {
        if (nLocal > 0) {
            forAllAttributes([&]<typename Attribute>(Attribute& attribute) {
                attribute->create(nLocal);
            });
            atomic_add(this->localNum_m, nLocal);
        }
    }
private:
    size_type newParticles_m
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform objects for output
        Inform msg("IndependentParticleTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();

        // Main Timer
        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);
        msg << "Independent Particles Test" << endl;

        // Particle Bunch Type
        using bunch_type =
            IndependentParticles<ippl::detail::ParticleLayout<double, Dim>, double, Dim>;
        std::unique_ptr<bunch_type> P;

        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(20.0);

        const double t_0 = 0.0;

        /* TODO: Define Proper Particle Layout
            Mesh_t<Dim> mesh(domain, hr, origin);
            FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
            PLayout_t<double, Dim> PL(FL, mesh);
        */

        // Initialize Particle Bunch
        ippl::detail::ParticleLayout<double, Dim> PL;
        P = std::make_unique<bunch_type>(PL, rmin, rmax);

        // Create Particles on each Rank
        size_type nloc = 1000;
        P->create(nloc);

        // Sample Particle Initial Locations
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();

        using RandomPool = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
        RandomPool rand_pool(12345);

        msg << "particles created and initial conditions assigned " << endl;

        // Begin Particle Tracks
        msg << "Starting Tracks..." << endl;
        Kokkos::parallel_for(P->getLocalNum(), 
            KOKKOS_LAMBDA(const int i){
                // NOTE: Dummy Work
                RandomPool::generator_type steps_generator = rand_pool.get_state();
                int steps = static_cast<int>(steps_generator.drand(0.0, 1.0) * 10000);
                volatile double dummy_result = 0.0; // Volatile variable
                for (int w = 0; w < steps; ++w) {
                    dummy_result += Kokkos::sin(static_cast<double>(w) * 0.01 + i * 0.001);
                }
                rand_pool.free_state(generator);
            }
        );
        
        // TODO: MPI WaitAll
        // TODO: Load Balancing
        // TODO: Restart Loop with new Particles

        msg << "Independent Particles Test: End." << endl;
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
