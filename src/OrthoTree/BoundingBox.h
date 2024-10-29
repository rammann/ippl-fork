#ifndef ORTHO_TREE_BOUNDS_GUARD
#define ORTHO_TREE_BOUNDS_GUARD

#include <sstream> // for to_string()
#include "Types.h"
namespace ippl {

    template <size_t Dim>
    class BoundingBox {
        using real_coordinate = real_coordinate_template<Dim>;
        using grid_coordinate = real_coordinate_template<Dim>;

        real_coordinate min_m;
        real_coordinate max_m;

    public:
        BoundingBox() = default;
        BoundingBox(real_coordinate min, real_coordinate max)
            : min_m(min), max_m(max)
        { }

        real_coordinate get_min() const { return min_m; }
        real_coordinate get_max() const { return max_m; }

        BoundingBox get_raster(size_t raster_size) const
        {
            BoundingBox raster;
            raster.min_m = min_m;
            for ( size_t i = 0; i < Dim; ++i ) {
                raster.max_m[i] = raster.min_m[i] + (max_m[i] - min_m[i]) / raster_size;
            }

            return raster;
        }

        real_coordinate get_center() const
        {
            real_coordinate center;
            for ( size_t i = 0; i < Dim; ++i ) {
                center[i] = min_m[i] + (max_m[i] - min_m[i]) / 2;
            }
            return center;
        }

        /**
         * @brief prints the bounding box to a string of the form: {(min),(max)}
         *
         * @return std::string
         */
        std::string to_string() const
        {
            // stringstream because ippl::Vector supports this
            std::stringstream res;
            res << "{min=" << min_m << ", max=" << max_m << "}";
            return res.str();
        }

        static BoundingBox bounds_from_grid_coord(const BoundingBox& root_bounds, const grid_coordinate& grid_coord, const size_t depth, const size_t max_depth)
        {
            BoundingBox result;

            // Total nodes per edge at the lowest level (leaf level)
            const double nodes_per_edge_max_depth = static_cast<double>(1 << max_depth);

            // Scaling factor to move from the finest level to the current depth
            const double nodes_per_edge_cur_depth = static_cast<double>(1 << depth);

            // Calculate the size of each node's edge at this depth level
            real_coordinate node_size = (root_bounds.max_m - root_bounds.min_m) / nodes_per_edge_max_depth;

            // Compute the minimum corner of the bounding box for this grid coordinate at the current depth
            result.min_m = root_bounds.min_m + (static_cast<real_coordinate>(grid_coord) * node_size);

            // The maximum corner is the minimum corner plus the node size
            result.max_m = result.min_m + (node_size * (nodes_per_edge_max_depth / nodes_per_edge_cur_depth));

            return result;
        }

    };

} // namespace ippl

#endif // ORTHO_TREE_BOUNDS_GUARD