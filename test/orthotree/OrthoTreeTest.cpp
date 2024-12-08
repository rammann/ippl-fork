#include "Ippl.h"

#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Vector.hpp>
#include <random>

#include "Utility/ParameterList.h"

#include "OrthoTree/OrthoTree.h"
#include "OrthoTree/helpers/BoundingBox.h"

int main(int argc, char* argv[])
{
    ippl::initialize(argc, argv);
    {
        static constexpr size_t Dim = 2;
        const size_t max_particles  = 10;
        const size_t max_depth      = 8;
        const size_t num_particles  = 5000;  // per processor

        const auto MIN_BOUND = 0.0;
        const auto MAX_BOUND = 1.0;

        ippl::OrthoTree<Dim> tree(
            max_depth, max_particles,
            ippl::BoundingBox<Dim>({MIN_BOUND, MIN_BOUND}, {MAX_BOUND, MAX_BOUND}));

        typedef ippl::ParticleSpatialLayout<double, Dim> playout_type;
        typedef ippl::OrthoTreeParticle<playout_type> bunch_type;

        playout_type PLayout;
        bunch_type bunch(PLayout);

        std::mt19937_64 eng;
        std::uniform_real_distribution<double> unif(MIN_BOUND, MAX_BOUND);

        bunch.create(num_particles);

        if (ippl::Comm->rank() == 0) {
            std::cout << "Created " << bunch.getTotalNum() << " particles, that is "
                      << num_particles << " per processor\n";
        }

        typename bunch_type::particle_position_type::HostMirror R_host = bunch.R.getHostMirror();

        // RANDOM INITIALISATION
        // typename bunch_type::rho_container_type::HostMirror RHO_host   =
        // bunch.rho.getHostMirror();
        /*
                for (unsigned int i = 0; i < num_particles; ++i) {
                    // R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng), unif(eng)};
                    R_host(i) = ippl::Vector<double, Dim>{unif(eng), unif(eng)};
                    //  RHO_host(i) = 0;
                }
        */

        // THIS GENERATES A SPIRAL DISTRIBUTION
        double armCount     = 2.0;   // Number of spiral arms
        double armTightness = -0.1;  // Tightness of the spiral arms
        double center_x     = (MAX_BOUND - MIN_BOUND) / 2;
        double center_y     = (MAX_BOUND - MIN_BOUND) / 2;
        double max_distance = 0.9 * (MAX_BOUND - MIN_BOUND) / 2;
        for (unsigned i = 0; i < num_particles; ++i) {
            double angle    = static_cast<double>(rand()) / RAND_MAX * 2.0 * M_PI;
            double distance = (static_cast<double>(rand()) / RAND_MAX) * max_distance;

            double totalArmAngle = 5.0;

            for (int j = 0; j < armCount; ++j) {
                // add spiral arm effect to the angle
                double armAngle =
                    armTightness * angle + (j + 1) * distance / max_distance * 2.0 * M_PI;
                double x = center_x + distance * cos(armAngle + totalArmAngle);
                double y = center_y + distance * sin(armAngle + totalArmAngle);

                R_host(i) = {x, y};
                totalArmAngle += armAngle;
            }
        }

        Kokkos::deep_copy(bunch.R.getView(), R_host);
        // Kokkos::deep_copy(bunch.rho.getView(), RHO_host);

        bunch.update();
        tree.build_tree(bunch);
        // tree.build_tree_naive(bunch);

        // stuff 1
        // tree.build_tree_naive(particles);

        // ippl::vector_t<ippl::morton_code> vec;
        // for ( ippl::morton_code c = 0; c < 100; ++c ) {
        //     vec.push_back(c);
        // }
        // tree.complete_tree(vec);
        // stuff//  2
        // tree.build_tree_naive(particles);

        /*
                auto lin_tree = (tree.get_tree());
                Kokkos::vector<ippl::morton_code> tree_copy(lin_tree.size());
                for ( size_t i = 0; i < lin_tree.size(); ++i ) {
                    tree_copy[i] = lin_tree[i].first;
                }

                tree.partition(tree_copy);
        */
    }

    ippl::finalize();

    return 0;
}
