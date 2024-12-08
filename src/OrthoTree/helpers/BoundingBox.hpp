#include <sstream>  // for to_string()

#include "BoundingBox.h"

namespace ippl {

    template <size_t Dim>
    BoundingBox<Dim> BoundingBox<Dim>::get_raster(size_t raster_size) const {
        BoundingBox raster;
        raster.min_m = min_m;
        for (size_t i = 0; i < Dim; ++i) {
            raster.max_m[i] = raster.min_m[i] + (max_m[i] - min_m[i]) / raster_size;
        }

        return raster;
    }

    template <size_t Dim>
    BoundingBox<Dim>::real_coordinate BoundingBox<Dim>::get_center() const {
        real_coordinate center;
        for (size_t i = 0; i < Dim; ++i) {
            center[i] = min_m[i] + (max_m[i] - min_m[i]) / 2;
        }
        return center;
    }

    template <size_t Dim>
    std::string BoundingBox<Dim>::to_string() const {
        // stringstream because ippl::Vector supports this and im lazy
        std::stringstream res;
        res << "{min=" << min_m << ", max=" << max_m << "}";
        return res.str();
    }

    template <size_t Dim>
    BoundingBox<Dim> BoundingBox<Dim>::bounds_from_grid_coord(
        const BoundingBox<Dim>& root_bounds, const BoundingBox<Dim>::grid_coordinate& grid_coord,
        const size_t depth, const size_t max_depth) {
        // Integer arithmetic for nodes per edge
        const size_t nodes_per_edge_cur_depth = 1u << depth;      // 2^depth
        const size_t nodes_per_edge_max_depth = 1u << max_depth;  // 2^max_depth

        // Calculate bounds size only once
        real_coordinate bounds_size = root_bounds.max_m - root_bounds.min_m;

        // Compute depth square size and unit square size directly
        real_coordinate depth_sq_size, unit_sq_size;
        for (size_t i = 0; i < Dim; ++i) {
            bounds_size[i]   = std::abs(bounds_size[i]);
            depth_sq_size[i] = bounds_size[i] / static_cast<double>(nodes_per_edge_cur_depth);
            unit_sq_size[i]  = bounds_size[i] / static_cast<double>(nodes_per_edge_max_depth);
        }

        // Compute min and max coordinates
        real_coordinate min_coord = root_bounds.min_m;
        for (size_t i = 0; i < Dim; ++i) {
            min_coord[i] += unit_sq_size[i] * static_cast<double>(grid_coord[i]);
        }

        real_coordinate max_coord = min_coord;
        for (size_t i = 0; i < Dim; ++i) {
            max_coord[i] += depth_sq_size[i];
        }

        // Validate the computed size
        real_coordinate computed_size = max_coord - min_coord;
        for (size_t i = 0; i < Dim; ++i) {
            assert(std::abs(computed_size[i] - depth_sq_size[i])
                   < 1e-6);  // Adjust tolerance as needed
        }

        // Return the bounding box for the node at this depth
        return BoundingBox(min_coord, max_coord);
    }

}  // namespace ippl