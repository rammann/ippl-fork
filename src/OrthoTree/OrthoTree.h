#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include "Types.h"
#include "OrthoTreeParticle.h"
#include "MortonHelper.h"
#include "BoundingBox.h"

#include <Kokkos_Vector.hpp>
#include <Kokkos_Pair.hpp>
namespace ippl {

    // this is defined outside of Types.h on purpose, as this is likely to change in the finalised implementation
    template <size_t Dim>
    using particle_type_template = OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>>;

    template <size_t Dim>
    class OrthoTree {
        using real_coordinate = real_coordinate_template<Dim>;
        using grid_coordinate = grid_coordinate_template<Dim>;
        using particle_type = particle_type_template<Dim>;
        using bounds_type = BoundingBox<Dim>;

        const size_t max_depth_m;
        const size_t max_particles_per_node_m;
        const bounds_type root_bounds_m;
        const Morton<Dim> morton_helper;

        Kokkos::vector<morton_code> tree;

    public:

        explicit OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_type& root_bounds)
            : max_depth_m(max_depth), max_particles_per_node_m(max_particles_per_node),
            root_bounds_m(root_bounds), morton_helper(max_depth)
        { }

        void build_tree_naive_sequential(particle_type const& particles)
        {
            const size_t n_particles = particles.getLocalNum();
            const size_t nodes_per_edge = (size_t(1) << max_depth_m);

            // insert the root into the tree
            tree.push_back(morton_code(0));

            Kokkos::vector<Kokkos::pair<morton_code, size_t>> aid_list(n_particles);
            for ( size_t i = 0; i < n_particles; ++i ) {
                grid_coordinate grid_coord;

                for ( size_t j = 0; j < Dim; ++j ) {
                    const double dist = particles.R(i)[j] - root_bounds_m.get_min()[j];
                    const double grid_len = root_bounds_m.get_max()[j] - root_bounds_m.get_min()[j];

                    const double ratio = dist / grid_len;
                    grid_coord[j] = static_cast<grid_t>(nodes_per_edge * ratio);
                }

                aid_list[i] = { morton_helper.encode(grid_coord, max_depth_m), i };
            }

            std::sort(aid_list.begin(), aid_list.end(), [ ] (const auto& a, const auto& b)
            {
                return a.second < b.second;
            });

            for ( const auto& [morton, id] : aid_list ) {
                std::cout << "(morton=" << morton << ", id=" << id << ")\n";
            }
        }

    };

} // namespace ippl

#endif // ORTHOTREE_GUARD
