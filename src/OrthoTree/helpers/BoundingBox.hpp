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
        // Calculate the number of nodes per edge at the current depth
        const double nodes_per_edge_cur_depth = static_cast<double>(1 << depth);
        const double nodes_per_edge_max_depth = static_cast<double>(1 << max_depth);
        real_coordinate bounds_size           = root_bounds.max_m - root_bounds.min_m;
        real_coordinate depth_sq_size         = bounds_size / nodes_per_edge_cur_depth;
        real_coordinate unit_sq_size          = bounds_size / nodes_per_edge_max_depth;

        for (size_t i = 0; i < Dim; ++i) {
            bounds_size[i]   = std::abs(bounds_size[i]);
            unit_sq_size[i]  = std::abs(unit_sq_size[i]);
            depth_sq_size[i] = std::abs(depth_sq_size[i]);
        }
        real_coordinate min_coord = root_bounds.min_m + (unit_sq_size * grid_coord);
        real_coordinate max_coord = min_coord + depth_sq_size;

        /*
        std::cout << "At depth: " << depth << "/" << max_depth << std::endl
            << "   " << "bounds size:  min" << root_bounds.get_min() << ", max: " <<
        root_bounds.get_max() << std::endl
            << "   " << "unit_sq_size: " << unit_sq_size << " (1/" << (1 << max_depth) << "th)" <<
        std::endl
            << "   " << "dept_sq_size: " << depth_sq_size << std::endl
            << "   " << "grid_coords:  " << grid_coord << " (out of " << nodes_per_edge_max_depth <<
        ")" <<  std::endl
            << "   " << "real_coords:  " << "min:" << min_coord << ", max:" << max_coord <<
        std::endl << std::endl;
        */

        auto test = max_coord - min_coord;

        for (size_t i = 0; i < Dim; ++i) {
            assert(std::abs(test[i] - depth_sq_size[i]) < 1e-6);  // Adjust tolerance if needed
        }

        // Return the bounding box for the node at this depth
        BoundingBox result(min_coord, max_coord);
        return result;
    }

}  // namespace ippl