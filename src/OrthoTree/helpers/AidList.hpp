#include "AidList.h"

namespace ippl {

    template <size_t Dim, typename PLayout>
    void AidList::initialize(const BoundingBox<Dim>& root_bounds, PLayout const& particles) {
        if (world_rank == 0) {
            initialize_from_rank<Dim>(max_depth, root_bounds, particles);
            logger << "Aid list is initialized with size: " << size() << endl;
        }
    }

    template <typename PLayout>
    bool AidList::is_gathered(ippl::ParticleBase<PLayout> const& particles) {
        return particles.getLocalNum() == particles.getTotalNum();
    }

    template <size_t Dim>
    void AidList::initialize_from_rank(
        size_t max_depth, const BoundingBox<Dim>& root_bounds,
        OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
        if (!this->is_gathered(particles)) {
            throw std::runtime_error(
                "can only initialize if all particles are gathered on one rank!");
        }

        const size_t n_particles               = particles.getTotalNum();
        const size_t grid_size                 = (size_t(1) << max_depth);
        const auto morton_helper               = Morton<Dim>(max_depth);
        using real_coordinate                  = real_coordinate_template<Dim>;
        using grid_coordinate                  = grid_coordinate_template<Dim>;
        const real_coordinate root_bounds_size = root_bounds.get_max() - root_bounds.get_min();

        // temporary structure to hold the data... TODO: make this smarter
        std::vector<std::pair<morton_code, size_t>> temp_aid_list(n_particles);

        for (size_t i = 0; i < n_particles; ++i) {
            const real_coordinate normalized =
                (particles.R(i) - root_bounds.get_min()) / root_bounds_size;

            const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * grid_size);
            temp_aid_list[i] = {morton_helper.encode(grid_coord, max_depth), i};

            std::cerr << "Coordinate: " << particles.R(i)
                      << " turns into: " << temp_aid_list[i].first
                      << " (coming from: " << grid_coord << ")" << std::endl;
        }

        // sort by morton codes: TODO: make this smarter
        std::sort(temp_aid_list.begin(), temp_aid_list.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        this->resize(n_particles);

        // ARE MIRRORS/DEEP COPIES REALLY NEEDED??
        auto host_octants      = Kokkos::create_mirror_view(this->getOctants());
        auto host_particle_ids = Kokkos::create_mirror_view(this->getParticleIDs());

        for (size_t i = 0; i < n_particles; ++i) {
            host_octants(i)      = temp_aid_list[i].first;
            host_particle_ids(i) = temp_aid_list[i].second;
        }

        Kokkos::deep_copy(this->getOctants(), host_octants);
        Kokkos::deep_copy(this->getParticleIDs(), host_particle_ids);
    }

}  // namespace ippl