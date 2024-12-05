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
        BoundingBox result;
        const double nodes_per_edge_max_depth = static_cast<double>(1 << max_depth);
        const double nodes_per_edge_cur_depth = static_cast<double>(1 << depth);
        real_coordinate node_size =
            (root_bounds.max_m - root_bounds.min_m) / nodes_per_edge_max_depth;
        result.min_m = root_bounds.min_m + (static_cast<real_coordinate>(grid_coord) * node_size);
        result.max_m =
            result.min_m + (node_size * (nodes_per_edge_max_depth / nodes_per_edge_cur_depth));

        return result;
    }

}  // namespace ippl