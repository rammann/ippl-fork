#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include "Types.h"
#include "OrthoTreeParticle.h"
#include "MortonHelper.h"
#include "BoundingBox.h"

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
        const size_t max_particles_per_node;
        const bounds_type root_bounds_m;
        const bounds_type rasterizer_m;

    public:

        explicit OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_type& root_bounds)
            : max_depth_m(max_depth), max_particles_per_node_m(max_particles_per_node),
            root_bounds_m(root_bounds)
        {
            const size_t leaf_nodes_per_edge = (size_t(1) << max_depth);
            rasterizer_m = root_bounds_m.get_raster(leaf_nodes_per_edge);
        }

        void build_tree_naive_sequential(particle_type const& particles);

    };

} // namespace ippl

#endif // ORTHOTREE_GUARD
