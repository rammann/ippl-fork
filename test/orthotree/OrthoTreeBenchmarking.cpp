#include "Ippl.h"

#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Vector.hpp>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "Utility/ParameterList.h"

#include "ArgParser.h"
#include "OrthoTree/OrthoTree.h"
#include "OrthoTree/helpers/BoundingBox.h"

using namespace ippl;

template <size_t Dim, typename ParticlePositioins>
void initializeSpiral(ParticlePositioins& particle_positions, const size_t num_particles);

template <size_t Dim, typename ParticlePositioins>
void initializeRandom(ParticlePositioins& particle_positions, const size_t num_particles);

template <size_t Dim>
auto initializeParticles(const size_t num_particles);

template <size_t Dim>
void run_experiment();

static void define_arguments() {
    // algorithm arguments
    ArgParser::add_argument<size_t>("dim", 2, "Dimension of the simulation");
    ArgParser::add_argument<size_t>("max_particles", 10, "Maximum particles per octant");
    ArgParser::add_argument<size_t>("max_depth", 8, "Maximum depth of the octree");
    ArgParser::add_argument<size_t>("num_particles_tot", 5000, "Number of particles in total");
    ArgParser::add_argument<double>("min_bounds", 0.0, "Min coordinate of the bounding box");
    ArgParser::add_argument<double>("max_bounds", 1.0, "Max coordinate of the bounding box");
    ArgParser::add_argument<size_t>("seed", std::random_device{}(),
        "Seed for the random initialisation, default is random");
    ArgParser::add_argument<std::string>("dist", "random",
        "Type of particle distribution, one of: {random, spiral}");

    // output arguments
    ArgParser::add_argument<std::string>("print_stats", "true",
        "Sets the log level for our outputs.");
    ArgParser::add_argument<std::string>("enable_visualisation", "false",
        "Enables or disables the output of visualisation data");
    ArgParser::add_argument<size_t>("log_level", 0, "Sets the log level for our outputs.");

    ArgParser::add_argument<std::string>("parallel", "true",
        "true for parallel, false for sequential run");
    ArgParser::add_argument<std::string>("visualize_helper", "false",
        "Enables the replicate this run message");
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        define_arguments();
        ArgParser::parse(argc, argv);
        const bool visualize_helper = ArgParser::get<bool>("visualize_helper");
        const size_t dimensions = ArgParser::get<size_t>("dim");

        // Add the for loop here
        for (int iteration = 0; iteration < 5; ++iteration) {
            // Optional: Print which iteration we're on
            if (Comm->rank() == 0) {
                std::cerr << "\nStarting iteration " << iteration + 1 << " of 5\n" << std::endl;
            }

            // logging at the beginning in case the run crashes
            if (Comm->rank() == 0 && visualize_helper) {
                std::cerr << "Replicate this run with: \n"
                    << "./visualise.sh " << Comm->size() << " " << ArgParser::get_args()
                    << std::endl;
            }

            // logging to find where it hangs
            if (Comm->rank() == 0) {
                std::cerr << "Running OrthoTree benchmark in " << dimensions << "D" << std::endl;
            }

            if (dimensions == 2) {
                run_experiment<2>();
            }
            else if (dimensions == 3) {
                run_experiment<3>();
            }
            else {
                std::cerr << "We only specialise for 2D and 3D!";
                exit(1);
            }

            // logging at the end in case the run crashes
            if (Comm->rank() == 0 && visualize_helper) {
                std::cerr << "Finished run with: \n"
                    << "./visualise.sh " << Comm->size() << " " << ArgParser::get_args()
                    << std::endl;
            }

            if (Comm->rank() == 0 && visualize_helper) {
                std::cerr << "Replicate this run with: \n"
                    << "./visualise.sh " << Comm->size() << " " << ArgParser::get_args()
                    << std::endl;
            }
            IpplTimings::print();
        }
    }
    ippl::finalize();
    return 0;
}


template <size_t Dim>
void run_experiment() {
    static_assert((Dim == 2 || Dim == 3) && "We only specialise for 2D and 3D!");

    const size_t max_particles = ArgParser::get<size_t>("max_particles");
    const size_t max_depth = ArgParser::get<size_t>("max_depth");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");

    ippl::OrthoTree<Dim> tree(
        max_depth, max_particles,
        ((Dim == 2) ? ippl::BoundingBox<Dim>({ min_bounds, min_bounds }, { max_bounds, max_bounds })
            : ippl::BoundingBox<Dim>({ min_bounds, min_bounds, min_bounds },
                { max_bounds, max_bounds, max_bounds })));

    const bool enable_visualisation = ArgParser::get<bool>("enable_visualisation");
    const size_t log_level = ArgParser::get<size_t>("log_level");
    const bool enable_stats = ArgParser::get<bool>("print_stats");
    const bool run_parallel = ArgParser::get<bool>("parallel");

    tree.setVisualisation(enable_visualisation);
    tree.setLogLevel(log_level);
    tree.setPrintStats(enable_stats);

    const size_t num_particles = ArgParser::get<size_t>("num_particles_tot");
    const size_t num_particles_per_proc = num_particles / Comm->size();
    auto particles = initializeParticles<Dim>(num_particles_per_proc);

    IpplTimings::TimerRef timer;
    timer = IpplTimings::getTimer("orthotree_build");
    IpplTimings::startTimer(timer);
    if (run_parallel)
        tree.build_tree(particles);
    else
        tree.build_tree_naive(particles);

    IpplTimings::stopTimer(timer);
    Comm->barrier();
}

template <size_t Dim>
auto initializeParticles(const size_t num_particles) {
    const std::string particle_distribution = ArgParser::get<std::string>("dist");

    typedef ippl::ParticleSpatialLayout<double, Dim> particle_layout_type;
    typedef ippl::OrthoTreeParticle<particle_layout_type> bunch_type;

    particle_layout_type PLayout;
    bunch_type bunch(PLayout);

    bunch.create(num_particles);

    typename bunch_type::particle_position_type::HostMirror positions_host =
        bunch.R.getHostMirror();

    if (particle_distribution == "spiral") {
        initializeSpiral<Dim>(positions_host, num_particles);
    }
    else if (particle_distribution == "random") {
        initializeRandom<Dim>(positions_host, num_particles);
    }
    else {
        std::cerr << "Distribution: " << particle_distribution << " is not suppoerted!";
        exit(1);
    }

    Kokkos::deep_copy(bunch.R.getView(), positions_host);

    bunch.update();
    return bunch;
}

template <size_t Dim, typename PLayout>
void initializeRandom(PLayout& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3) && "We only specialise for 2D and 3D!");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const size_t seed = ArgParser::get<size_t>("seed");

    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bounds, max_bounds);

    for (unsigned int i = 0; i < num_particles; ++i) {
        if constexpr (Dim == 2) {
            particle_positions(i) = ippl::Vector<double, Dim>{ unif(eng), unif(eng) };
        }
        else if constexpr (Dim == 3) {
            particle_positions(i) = ippl::Vector<double, Dim>{ unif(eng), unif(eng), unif(eng) };
        }
        else {
            std::cerr << "We only specialise for 2D and 3D!" << std::endl;
            exit(1);
        }
    }
}

template <size_t Dim, typename ParticlePositioins>
void initializeSpiral(ParticlePositioins& particle_positions, const size_t num_particles) {
    static_assert((Dim == 2 || Dim == 3) && "Spirals are only possible in 2D or 3D!");

    const double min_bounds = ArgParser::get<double>("min_bounds");
    const double max_bounds = ArgParser::get<double>("max_bounds");
    const double bounds_size = max_bounds - min_bounds;

    const size_t seed = ArgParser::get<size_t>("seed");
    std::mt19937_64 eng(seed);
    std::uniform_real_distribution<double> unif(min_bounds, max_bounds);

    // center of the bounding box
    const double center_x = bounds_size / 2;
    const double center_y = bounds_size / 2;
    const double center_z = (Dim == 3) ? bounds_size / 2 : 0.0;

    double armCount = 2.0;   // number of spiral arms
    double armTightness = -0.1;  // tightness of the spiral arms

    // max distance from the center
    double max_distance = 0.9 * bounds_size / 2;

    for (unsigned i = 0; i < num_particles; ++i) {
        double angle = unif(eng) * 2.0 * M_PI;
        double distance = unif(eng) * max_distance;

        double totalArmAngle = 5.0;

        for (int j = 0; j < armCount; ++j) {
            double armAngle = armTightness * angle + (j + 1) * distance / max_distance * 2.0 * M_PI;
            double x = center_x + distance * cos(armAngle + totalArmAngle);
            double y = center_y + distance * sin(armAngle + totalArmAngle);
            double z = (Dim == 3) ? center_z + distance * sin(angle) : 0.0;

            if constexpr (Dim == 2) {
                particle_positions(i) = { x, y };
            }
            else if constexpr (Dim == 3) {
                particle_positions(i) = { x, y, z };
            }

            totalArmAngle += armAngle;
        }
    }
}