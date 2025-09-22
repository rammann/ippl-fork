// Particle Container Test
// Usage:
//          srun ./ParticleContainerTest
//              <nt> <ppsp> --info 5
//
//          nt    = No. timesteps 
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
#include "Types/IpplTypes.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleLayout.h"
#include "Region/NDRegion.h"


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

using bool_type   = typename ippl::detail::ViewType<bool, 1>::view_type;

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

// random int generator
template <typename T, class GeneratorPool>
struct generate_random_int {
    using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type vals;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T start, end;

    // Initialize all members
    generate_random_int(view_type vals_, GeneratorPool rand_pool_)
        : vals(vals_)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        // Draw 0 or 1
        vals(i) = rand_gen.rand_int(2);

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};


// Particle Layout, since we don't have Fields and don't need ParticleSpatialLayout
template <typename T, unsigned Dim, typename... PositionProperties>
class SecondaryParticleLayout : public ippl::detail::ParticleLayout<T, Dim, PositionProperties...>{
public:
    SecondaryParticleLayout() : ippl::detail::ParticleLayout<T, Dim, PositionProperties...>() {}
    ~SecondaryParticleLayout() = default;

public:
    void setDomain(ippl::NDRegion<T,Dim> domain){
        domain_m=domain;
    }
    template <class ParticleContainer>
    void update(ParticleContainer& pc){
        // Apply BC
        this->applyBC(pc.R,domain_m);
    }

private:
    ippl::NDRegion<T,Dim> domain_m;
    
};


// Particle Containers
// 1. Species given by attribute
template <class PLayout, typename T, unsigned Dim = 3>
class SecondaryParticleContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    typename Base::particle_position_type P; // Momenta
    ParticleAttrib<int> Sp; // Particle Species 

    SecondaryParticleContainer(PLayout& pl)
        : Base(pl)
    {
        this->addAttribute(P);
        this->addAttribute(Sp);
        rand_pool64=Kokkos::Random_XorShift64_Pool<>((size_type)(42 + 100 * ippl::Comm->rank()));
    }

    ~SecondaryParticleContainer() {}

    
    void initialize(Vector_t<T,Dim> rmin_, Vector_t<T,Dim> rmax_){
        //Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        rmin=rmin_;
        rmax=rmax_;
        Kokkos::parallel_for(
            this->getLocalNum(), generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      this->R.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();
        
        // Sample use same domain for P as for R
        Kokkos::parallel_for(
            this->getLocalNum(), generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      this->P.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();
                
        // Randomly sample species
        Kokkos::fill_random(this->Sp.getView(), rand_pool64,2);
    }

    void getMuonMomenta(){
        Kokkos::parallel_for(
        this->getLocalNum(), 
            generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>
            (this->P.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();
    }

    void getElectronMomenta(){
        Kokkos::parallel_for(
            this->getLocalNum(), 
            generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>
            (this->P.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();
        
    }

    void muonDrift(double dt){
        Kokkos::parallel_for(this->getLocalNum(), 
        [=,this](const int i){
            if(this->Sp(i)==0){
                this->R(i) = this->R(i) + dt * this->P(i);
            }
        });
        Kokkos::fence();
    }

    void electronDrift(double dt){
        Kokkos::parallel_for(this->getLocalNum(), 
        [=,this](const int i){
            if(this->Sp(i)==1){
                this->R(i) = this->R(i) + dt * this->P(i);
            }
        });
        Kokkos::fence();
    }

    void muonDecay(){
        Kokkos::parallel_for(this->getLocalNum(),
        [=,this](const int i){
            if(this->Sp(i)==0){
                auto rand_gen = rand_pool64.get_state();
                if(rand_gen.drand(0.0,1.0)<0.1){
                    this->Sp(i)=1;
                    this->P(i) = this->P(i)/2;
                }
                rand_pool64.free_state(rand_gen);
            }
        });
        Kokkos::fence();
    }

    void ionization(){
        int new_particles;
        Kokkos::parallel_reduce(this->getLocalNum(),
        [=,this](const int& i, int& sum){
            if(this->Sp(i)==1){
                auto rand_gen = rand_pool64.get_state();
                if(rand_gen.drand(0.0,1.0)<0.1){
                    sum += 1;
                }
                rand_pool64.free_state(rand_gen);
            }
        }, new_particles);
        Kokkos::fence();
        this->create(new_particles);

        // Set species
        Kokkos::parallel_for(Kokkos::RangePolicy(this->getLocalNum()-new_particles-1, this->getLocalNum()),
        [=,this](const int i){
            this->Sp(i)=0;
        });
        
        // Sample positions and momenta
        Kokkos::parallel_for(
            Kokkos::RangePolicy(this->getLocalNum()-new_particles-1, this->getLocalNum()), 
            generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                    this->R.getView(), rand_pool64, rmin, rmax));
        Kokkos::parallel_for(
            Kokkos::RangePolicy(this->getLocalNum()-new_particles-1, this->getLocalNum()), 
            generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                    this->P.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();
    }

    void particleLoss(){
        bool_type lostParticles("lost particles", this->getLocalNum());
        int nLost;
        Kokkos::parallel_reduce(this->getLocalNum(),
        [=,this](const int& i, int&sum){
            auto rand_gen = rand_pool64.get_state(); 
            if(rand_gen.drand(0.0,1.0)<0.05){
                lostParticles(i)=1;
                sum += 1;
            }
        },nLost);
        this->destroy(lostParticles, nLost);
 
    }

private:
    Kokkos::Random_XorShift64_Pool<> rand_pool64; 
    Vector_t<double, Dim> rmin;
    Vector_t<double, Dim> rmax;
};

// 2. Species given by sorted containter
// Assuming only two species for simplicity
template <class PLayout, typename T, unsigned Dim = 3>
class SortedParticleContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    typename Base::particle_position_type P; // Momenta
    
    SortedParticleContainer(PLayout& pl)
        : Base(pl)
    {
        this->addAttribute(P);
    }

    ~SortedParticleContainer() {}
    
    // Overwrite the ParticleBase::create(...) function so we can set first_idx
    void create(unsigned int nsp, size_type ppsp){
        assert(nsp==2);
        first_idx=ppsp;
        Base::create(nsp*ppsp);
    }

private:
    // index of the first particle of the second species
    unsigned int first_idx;

};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Message objects
        Inform msg("ParticleContainerTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        // Timers
        auto start = std::chrono::high_resolution_clock::now();
        static IpplTimings::TimerRef mainTimer          = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef muonMomenta        = IpplTimings::getTimer("muonMomenta");
        static IpplTimings::TimerRef electronMomenta    = IpplTimings::getTimer("electronMomenta");
        static IpplTimings::TimerRef muonDrift          = IpplTimings::getTimer("muonDrift");
        static IpplTimings::TimerRef electronDrift      = IpplTimings::getTimer("electronDrift");
        static IpplTimings::TimerRef muonDecay          = IpplTimings::getTimer("muonDecay");
        static IpplTimings::TimerRef ionization         = IpplTimings::getTimer("ionization");
        static IpplTimings::TimerRef particleLoss         = IpplTimings::getTimer("particleLoss");

        IpplTimings::startTimer(mainTimer);
       
        // Read inputs
        
        int arg=1;
        const unsigned int nt   = std::atoi(argv[arg++]);
        //unsigned int nSp        = std::atoi(argv[arg++]); 
        size_type ppSp          = std::atoll(argv[arg++]);
        const double dt              = 1.0;
        msg << "Particle Container Test" << endl
            << "nt " << nt << " ppSp = " << ppSp << endl;

        // Define Domain
        Vector_t<double, Dim> rmin(-1.0);
        Vector_t<double, Dim> rmax(1.0); 
        ippl::NDRegion<double,3> domain;
        for(unsigned int d=0;d<Dim;++d){
            domain[d] = ippl::PRegion<double>(-1,1);
        }
        
        // Particle Layout
        SecondaryParticleLayout<double,Dim> PL;
        PL.setDomain(domain);
        PL.setParticleBC(ippl::BC::PERIODIC);

        // Particle Container Pointer
        using container_type = SecondaryParticleContainer<SecondaryParticleLayout<double,Dim>, double, Dim>;
        std::shared_ptr<container_type> PC = std::make_shared<container_type>(PL);

        // Create particles
        PC->create(2*ppSp);

        // Initialize particle species, positions and momenta
        PC->initialize(rmin, rmax);

        // Initialize RNG
        //Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));

        // Simulation Loop
        msg << "Starting iterations..." << endl;
        for(unsigned int it=0; it<nt; it++){

            //  Calculate Muon momenta
            IpplTimings::startTimer(muonMomenta);
            PC->getMuonMomenta(); 
            IpplTimings::stopTimer(muonMomenta);
        
            // Muon drift
            IpplTimings::startTimer(muonDrift);
            PC->muonDrift(dt); 
            IpplTimings::stopTimer(muonDrift);
            
            //  Calculate Electron momenta
            IpplTimings::startTimer(electronMomenta);
            PC->getElectronMomenta();
            IpplTimings::stopTimer(electronMomenta); 
            
            // Electron drift
            IpplTimings::startTimer(electronDrift);
            PC->electronDrift(dt); 
            IpplTimings::stopTimer(electronDrift);

            // Apply BC
            PC->update();

            // Muon -> Electron Decay
            IpplTimings::startTimer(muonDecay);
            PC->muonDecay(); 
            IpplTimings::stopTimer(muonDecay);

            // Electron births through ionization
            IpplTimings::startTimer(ionization);
            PC->ionization();
            IpplTimings::stopTimer(ionization);
            
            // Particle losses
            IpplTimings::startTimer(particleLoss);
            PC->particleLoss();
            IpplTimings::stopTimer(particleLoss);
            
        }
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
