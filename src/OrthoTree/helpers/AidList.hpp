#include "AidList.h"

namespace ippl {
    template <size_t Dim>
    AidList<Dim>::AidList(size_t max_depth)
        : world_rank(Comm->rank())
        , world_size(Comm->size())
        , max_depth(max_depth)
        , morton_helper(Morton<Dim>(max_depth))
        , logger("AidList", std::cerr, INFORM_ALL_NODES) {
        logger.setOutputLevel(5);
        logger.setPrintNode(INFORM_ALL_NODES);

        logger << "Initialized AidList" << endl;
    }

    template <size_t Dim>
    template <typename PLayout>
    void AidList<Dim>::initialize(const BoundingBox<Dim>& root_bounds, PLayout const& particles) {
        if (world_rank == 0) {
            initialize_from_rank(max_depth, root_bounds, particles);
            logger << "Aid list is initialized with size: " << size() << endl;
        }
    }

    template <size_t Dim>
    template <typename PLayout>
    bool AidList<Dim>::is_gathered(ippl::ParticleBase<PLayout> const& particles) {
        return particles.getLocalNum() == particles.getTotalNum();
    }

    template <size_t Dim>
    void AidList<Dim>::initialize_from_rank(
        size_t max_depth, const BoundingBox<Dim>& root_bounds,
        OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
        if (world_rank != 0) {
            throw std::runtime_error("kys aogfbeuebgueigbewiugbwlgwegbewkjl");
        }

        if (!this->is_gathered(particles)) {
            throw std::runtime_error(
                "can only initialize if all particles are gathered on one rank!");
        }

        const size_t n_particles               = particles.getTotalNum();
        const size_t grid_size                 = (size_t(1) << max_depth);
        using real_coordinate                  = real_coordinate_template<Dim>;
        using grid_coordinate                  = grid_coordinate_template<Dim>;
        const real_coordinate root_bounds_size = root_bounds.get_max() - root_bounds.get_min();

        // temporary structure to hold the data... TODO: make this smarter
        std::vector<std::pair<morton_code, size_t>> temp_aid_list(n_particles);

        for (size_t i = 0; i < n_particles; ++i) {
            const real_coordinate normalized =
                (particles.R(i) - root_bounds.get_min()) / root_bounds_size;

            const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * grid_size);
            temp_aid_list[i]                 = {morton_helper.encode(grid_coord, max_depth), i};
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

    template <size_t Dim>
    std::pair<morton_code, morton_code> AidList<Dim>::getMinReqOctants() {
        static constexpr size_t view_size = 2;
        Kokkos::View<morton_code[view_size]> min_max_octants("min_max_octants");
        if (world_rank == 0) {
            const size_t size        = this->size();
            const size_t batch_size  = size / world_size;
            const size_t rank_0_size = batch_size + (size % batch_size);

            Kokkos::parallel_for(
                "Send min/max octants", world_size - 1, [=, this](const size_t target_rank) {
                    const size_t start = ((target_rank + 1) * batch_size) + rank_0_size;
                    const size_t end   = start + batch_size - 1;

                    min_max_octants(0) = getOctant(start);
                    min_max_octants(1) = getOctant(end);

                    Comm->send(*min_max_octants.data(), view_size, target_rank + 1, 0);
                });

            // assign min/max for rank 0
            min_max_octants(0) = getOctant(0);
            min_max_octants(1) = getOctant(rank_0_size - 1);
        } else {
            mpi::Status status;
            try {
                Comm->recv(min_max_octants.data(), view_size, 0, 0, status);
            } catch (IpplException& e) {
                std::cerr << "ERROR in recv: " << e.what() << std::endl;
            }
        }

        auto host_min_max = Kokkos::create_mirror_view(min_max_octants);
        Kokkos::deep_copy(host_min_max, min_max_octants);

        return std::make_pair(host_min_max(0), host_min_max(1));
    }

    template <size_t Dim>
    size_t AidList<Dim>::getLowerBoundIndex(morton_code octant) const {
        size_t lower_bound = 0;

        // chatgpt magic
        Kokkos::parallel_reduce(
            "LowerBoundSearch", this->size(),
            [=, this](const size_t i, size_t& update) {
                if (octants(i) >= octant) {
                    update = (update < i) ? update : i;
                }
            },
            Kokkos::Min<size_t>(lower_bound));

        if (lower_bound == this->size()) {
            throw std::runtime_error("Octant not found in AidList.");
        }
        return lower_bound;
    }
    template <size_t Dim>
    size_t AidList<Dim>::getUpperBoundIndexExclusive(morton_code octant) const {
        size_t upper_bound = this->size();

        Kokkos::parallel_reduce(
            "UpperBoundExclusiveSearch", this->size(),
            [=, this](const size_t i, size_t& update) {
                if (octants(i) > octant) {
                    update = (update < i) ? update : i;
                }
            },
            Kokkos::Min<size_t>(upper_bound));

        return upper_bound;
    }

    template <size_t Dim>
    size_t AidList<Dim>::getUpperBoundIndexInclusive(morton_code octant) const {
        size_t upper_bound = this->size();

        Kokkos::parallel_reduce(
            "UpperBoundInclusiveSearch", this->size(),
            [=, this](const size_t i, size_t& update) {
                if (octants(i) >= octant) {
                    update = (update < i) ? update : i;
                }
            },
            Kokkos::Min<size_t>(upper_bound));

        if (upper_bound == this->size()) {
            return upper_bound - 1;
        }

        return upper_bound;
    }

    template <size_t Dim>
    size_t AidList<Dim>::getNumParticlesInOctant(morton_code octant) const {
        const size_t lower_bound_idx =
            getLowerBoundIndex(morton_helper.get_deepest_first_descendant(octant));
        const size_t upper_bound_idx =
            getUpperBoundIndexInclusive(morton_helper.get_deepest_last_descendant(octant));

        if (lower_bound_idx > upper_bound_idx) {
            throw std::runtime_error("loweridx > upper_idx in getNumParticlesInOctant...");
        }

        return upper_bound_idx - lower_bound_idx;
    }
}  // namespace ippl