#include "AidList.h"

namespace ippl {

    template <size_t Dim>
    bool AidList::is_gathered(
        OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
        if (world_rank != 0) {
            throw std::runtime_error("only call AidList::is_gathered from rank 0!");
        }

        return particles.getLocalNum() == particles.getTotalNum();
    }

    template <size_t Dim>
    void AidList::initialize_from_rank_0(
        size_t max_depth, const BoundingBox<Dim>& root_bounds,
        OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
        if (world_rank != 0) {
            return;
        }

        if (particles.getLocalNum() != particles.getTotalNum()) {
            throw std::runtime_error(
                "can only initialize if all particles are gathered on rank 0!");
        }

        const size_t n_particles = particles.getLocalNum();
        const size_t grid_size   = (size_t(1) << max_depth);

        using real_coordinate                  = real_coordinate_template<Dim>;
        using grid_coordinate                  = grid_coordinate_template<Dim>;
        const auto morton_helper               = Morton<Dim>(max_depth);
        const real_coordinate root_bounds_size = root_bounds.get_max() - root_bounds.get_min();

        // Temporary structure to hold pairs of Morton code and particle index for sorting
        std::vector<std::pair<size_t, size_t>> temp_aid_list(n_particles);

        // Populate the temporary vector
        for (size_t i = 0; i < n_particles; ++i) {
            const real_coordinate normalized =
                (particles.R(i) - root_bounds.get_min()) / root_bounds_size;

            const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * grid_size);

            temp_aid_list[i] = {morton_helper.encode(grid_coord, max_depth), i};
        }

        // Sort the temporary vector by Morton codes
        std::sort(temp_aid_list.begin(), temp_aid_list.end());

        // Allocate Kokkos View and copy sorted data back
        this->aid_list_m   = Kokkos::View<size_t* [2]>("aid_list_m", n_particles);
        auto host_aid_list = Kokkos::create_mirror_view(aid_list_m);

        for (size_t i = 0; i < n_particles; ++i) {
            host_aid_list(i, 0) = temp_aid_list[i].first;   // Morton code
            host_aid_list(i, 1) = temp_aid_list[i].second;  // Particle index
        }

        // Copy the sorted data back to the device
        Kokkos::deep_copy(aid_list_m, host_aid_list);
    }

}  // namespace ippl