#ifndef ORTHO_TREE_BOUNDS_GUARD
#define ORTHO_TREE_BOUNDS_GUARD

#include "Types.h"

namespace ippl {

    template <size_t Dim>
    class BoundingBox {
        using real_coordinate = real_coordinate_template<Dim>;

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
                raster.max_m[i] = raster.min_m + (max_m - min_m) / raster_size;
            }

            return raster;
        }

        real_coordinate get_center() const
        {
            real_coordinate center;
            for ( size_t i = 0; i < Dim; ++i ) {
                center[i] = min_m + (max_m - min_m) / 2;
            }
            return center;
        }
    };

} // namespace ippl

#endif // ORTHO_TREE_BOUNDS_GUARD