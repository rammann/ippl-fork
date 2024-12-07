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
            // this gets rid of cancellation, thank you @NumCSE script
            const grid_coordinate grid_coord = static_cast<grid_coordinate>(
                (particles.R(i) - root_bounds.get_min()) * (grid_size - 1) / root_bounds_size);

            // encode the grid coordinate and store it
            temp_aid_list[i] = {morton_helper.encode(grid_coord, max_depth), i};
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
    size_t AidList<Dim>::getLowerBoundIndex(morton_code target_octant) const {
        const auto lower_bound_it =
            std::lower_bound(octants.data(), octants.data() + octants.extent(0), target_octant,
                             [](const morton_code& octants_entry, const morton_code& target) {
                                 return octants_entry < target;
                             });

        return static_cast<size_t>(lower_bound_it - this->octants.data());
    }

    template <size_t Dim>
    size_t AidList<Dim>::getUpperBoundIndexExclusive(morton_code target_octant) const {
        const auto upper_bound_it =
            std::upper_bound(octants.data(), octants.data() + octants.extent(0), target_octant,
                             [](const morton_code& target, const morton_code& octants_entry) {
                                 return target < octants_entry;
                             });

        return static_cast<size_t>(upper_bound_it - this->octants.data());
    }

    template <size_t Dim>
    size_t AidList<Dim>::getUpperBoundIndexInclusive(morton_code target_octant) const {
        const size_t upper_bound_idx = getUpperBoundIndexExclusive(target_octant);
        return upper_bound_idx != 0 ? upper_bound_idx - 1 : upper_bound_idx;
    }

    template <size_t Dim>
    size_t AidList<Dim>::getNumParticlesInOctant(morton_code octant) const {
        const size_t lower_bound_idx = getLowerBoundIndex(octant);
        const size_t upper_bound_idx =
            getUpperBoundIndexExclusive(morton_helper.get_deepest_last_descendant(octant));

        if (lower_bound_idx > upper_bound_idx) {
            throw std::runtime_error("loweridx > upper_idx in getNumParticlesInOctant...");
        }

        return upper_bound_idx - lower_bound_idx;
    }

    template <size_t Dim>
    std::pair<morton_code, morton_code> AidList<Dim>::getMinReqOctants() {
        morton_code min_max_octants[2];
        if (world_rank == 0) {
            const size_t size       = this->size();
            const size_t batch_size = size / world_size;
            const size_t remainder  = (size % batch_size) + batch_size;

            // rank_{i+1} = [ (i * batch_size) + remainder, (i+1) * batch_size ]
            Kokkos::parallel_for("Send min/max octants", world_size - 1, [=, this](const size_t i) {
                morton_code
                    local_min_max_octants[2];  // inside the parallel for because of read-only
                const size_t start = (i * batch_size) + remainder;
                const size_t end   = start + batch_size;

                local_min_max_octants[0] = getOctant(start);
                local_min_max_octants[1] = getOctant(end - 1);

                // send to rank + 1, as parallel_for starts at index=0
                const size_t target_rank = i + 1;
                Comm->send(*local_min_max_octants, 2, target_rank, 0);
            });

            // assign min/max for rank 0
            // rank_0 = [ (#ranks - 1) * batch_size, #octants ]

            const size_t start = 0;
            const size_t end   = start + remainder;

            min_max_octants[0] = getOctant(start);
            min_max_octants[1] = getOctant(end - 1);
        } else {
            mpi::Status status;
            Comm->recv(min_max_octants, 2, 0, 0, status);
        }

        return std::make_pair(min_max_octants[0], min_max_octants[1]);
    }

    template <size_t Dim>
    void AidList<Dim>::innitFromOctants(morton_code min_octant, morton_code max_octant) {
        size_t size_buff;
        morton_code min_max_buff[2];
        if (world_rank == 0) {
            for (size_t rank = 1; rank < world_size; ++rank) {
                mpi::Status status;
                Comm->recv(*min_max_buff, 2, rank, 0, status);

                const size_t start = getLowerBoundIndex(min_max_buff[0]);
                const size_t end   = getUpperBoundIndexExclusive(min_max_buff[1]);
                size_buff          = end - start;

                Comm->send(size_buff, 1, rank, 0);

                Comm->send(*(octants.data() + start), size_buff, rank, 0);
                Comm->send(*(particle_ids.data() + start), size_buff, rank, 0);

                logger << "sent " << size_buff << " octants to rank " << rank << " range: ("
                       << getOctant(start) << ", " << getOctant(end) << ")" << endl;
            }
        } else {
            min_max_buff[0] = min_octant;
            min_max_buff[1] = morton_helper.get_deepest_last_descendant(max_octant);

            Comm->send(*min_max_buff, 2, 0, 0);

            mpi::Status size_status;
            Comm->recv(&size_buff, 1, 0, 0, size_status);

            this->octants      = Kokkos::View<morton_code*>("aid_list_octants", size_buff);
            this->particle_ids = Kokkos::View<size_t*>("aid_list_particle_ids", size_buff);

            mpi::Status octants_status, particle_id_status;
            Comm->recv(octants.data(), size_buff, 0, 0, octants_status);
            Comm->recv(particle_ids.data(), size_buff, 0, 0, particle_id_status);
        }
    }

    template <size_t Dim>
    template <typename Iterator>
    Kokkos::vector<size_t> AidList<Dim>::getNumParticlesInOctantsParalell(Iterator begin,
                                                                          Iterator end) {
        size_t size_buff;

        Kokkos::vector<size_t> weights;
        if (world_rank == 0) {
            for (size_t rank = 1; rank < world_size; ++rank) {
                weights.clear();
                mpi::Status size_status;
                Comm->recv(&size_buff, 1, rank, 0, size_status);
                std::vector<morton_code> octants_buff(size_buff);
                mpi::Status octants_status;
                Comm->recv(octants.data(), size_buff, rank, 0, octants_status);

                for (auto octant : octants_buff) {
                    weights.push_back(getNumParticlesInOctant(octant));
                }

                Comm->send(*weights.data(), size_buff, rank, 0);
            }

            weights.clear();
            for (auto it = begin; it != end; ++it) {
                weights.push_back(getNumParticlesInOctant(*it));
            }

        } else {
            size_buff = static_cast<size_t>(end - begin);
            Comm->send(size_buff, 1, 0, 0);
            Comm->send(*begin, size_buff, 0, 0);
            weights.resize(size_buff);
            mpi::Status weights_status;
            Comm->recv(weights.data(), size_buff, 0, 0, weights_status);
        }

        return weights;
    }
}  // namespace ippl