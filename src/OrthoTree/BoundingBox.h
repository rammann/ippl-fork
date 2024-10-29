#ifndef ORTHO_TREE_BOUNDS_GUARD
#define ORTHO_TREE_BOUNDS_GUARD

#include "OrthoTreeTypes.h"

namespace ippl {

    /**
     * @brief This class manages bounds of our domain. As i understand it we only care about the root bounds.
     * Most of this class is hence not used, but its nice to have.
     *
     * Can also be used to convert a grid coordinate back to a real valued BoundingBox.
     *
     */
    template <size_t Dim>
    class BoundingBox {
        using real_coordinate = real_coordinate_template<Dim>;
        using grid_coordinate = real_coordinate_template<Dim>;

        const real_coordinate min_m;
        const real_coordinate max_m;

    public:

        /**
         * @brief Construct a new BoundingBox with coordinates {0},{0} in R^Dim
         */
        BoundingBox() = default;

        /**
         * @brief Construct a new BoundingBox with fixed coordinates
         */
        BoundingBox(real_coordinate min, real_coordinate max)
            : min_m(min), max_m(max)
        { }

        real_coordinate get_min() const { return min_m; }
        real_coordinate get_max() const { return max_m; }

        /**
         * @brief Calculates the raster of this bounding box based on a raster size
         *
         * @param raster_size = 2^max_depth
         * @return BoundingBox
         */
        BoundingBox get_raster(size_t raster_size) const;

        /**
         * @brief Get the center of this bounding box
         *
         * @return real_coordinate center of thix box
         */
        real_coordinate get_center() const;

        /**
         * @brief prints the bounding box to a string of the form: {(min),(max)}
         *
         * @return std::string
         */
        std::string to_string() const;

        /**
         * @brief Calculate the real valued bounding box based on a coordinate (anchor) in the grid.
         * This is not the final version.
         *
         * @param root_bounds The root bounding box
         * @param grid_coord The anchor of the node we want to convert
         * @param depth The depth of the node we want to convert
         * @param max_depth The max depth of our tree
         * @return BoundingBox
         */
        static BoundingBox bounds_from_grid_coord(const BoundingBox& root_bounds, const grid_coordinate& grid_coord, const size_t depth, const size_t max_depth);

    };

} // namespace ippl

#include "BoundingBox.hpp"

#endif // ORTHO_TREE_BOUNDS_GUARD