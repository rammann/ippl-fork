#include "Ippl.h"
#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>

#include <random>
#include "Utility/ParameterList.h"
#include "Utility/IpplTimings.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {

    static auto timer = IpplTimings::getTimer("Orthotree Poisson Solver");
    
    typedef ippl::ParticleSpatialLayout<double, 3> playout_type;
    playout_type PLayout;

    // Targets
    ippl::OrthoTreeParticle targets(PLayout);
    unsigned int nTargets = 200;
    targets.create(nTargets);

    // Sources
    ippl::OrthoTreeParticle sources(PLayout);
    unsigned int nSources = 200;
    sources.create(nSources);

    // Random generators for position and charge
    std::mt19937_64 eng(42);
    std::uniform_real_distribution<double> posDis(0.0, 1.0);
    std::uniform_real_distribution<double> chargeDis(-20,20);

    // Generate target points
    
    for(unsigned int idx=0; idx<nTargets; ++idx){
        ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
        targets.R(idx) = r;
        targets.rho(idx) = 0.0;
    }

    // Generate source points
    for(unsigned int idx=0; idx<nSources; ++idx){
        ippl::Vector<double,3> r = {posDis(eng), posDis(eng), posDis(eng)};
        sources.R(idx) = r;
        sources.rho(idx) = chargeDis(eng);
    }
    
    // Tree Params
    ippl::ParameterList treeparams;
    treeparams.add("maxdepth",          7);
    treeparams.add("maxleafelements",   20);
    treeparams.add("boxmin",            0.0);
    treeparams.add("boxmax",            1.0);
    treeparams.add("sourceidx",         nTargets);

    // Solver Params
    ippl::ParameterList solverparams;
    solverparams.add("eps", 0.000001);

    
    ippl::TreeOpenPoissonSolver solver(targets, sources, treeparams, solverparams);

    IpplTimings::startTimer(timer);
    solver.Solve();
    IpplTimings::stopTimer(timer);

    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));

    
    }
    ippl::finalize();
}