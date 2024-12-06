#include "AidList.h"

namespace ippl {
    AidList::AidList(size_t max_depth)
        : world_rank(Comm->rank())
        , world_size(Comm->size())
        , max_depth(max_depth)
        , logger("AidList", std::cerr, INFORM_ALL_NODES) {
        logger.setOutputLevel(5);
        logger.setPrintNode(INFORM_ALL_NODES);

        logger << "Initialized AidList" << endl;
    }

    std::pair<morton_code, morton_code> AidList::getMinReqOctants() {
        static constexpr size_t view_size = 2;
        Kokkos::View<morton_code[view_size]> min_max_octants("min_max_octants");
        if (world_rank == 0) {
            const size_t size        = this->size();
            const size_t batch_size  = size / world_size;
            const size_t rank_0_size = batch_size + (size % batch_size);

            Kokkos::parallel_for(
                "Send min/max octants", world_size - 1, [=, this](const size_t target_rank) {
                    const size_t start = ((target_rank - 1) * batch_size) + rank_0_size;
                    const size_t end   = start + batch_size - 1;

                    min_max_octants(0) = getOctant(start);
                    min_max_octants(1) = getOctant(end);

                    Comm->send(*min_max_octants.data(), view_size, target_rank, 0);
                });

            // assign min/max for rank 0
            min_max_octants(0) = getOctant(0);
            min_max_octants(1) = getOctant(rank_0_size - 1);
        } else {
            mpi::Status status;
            Comm->recv(min_max_octants.data(), view_size, 0, 0, status);
        }

        auto host_min_max = Kokkos::create_mirror_view(min_max_octants);
        Kokkos::deep_copy(host_min_max, min_max_octants);

        return std::make_pair(host_min_max(0), host_min_max(1));
    }

    size_t AidList::getLowerBoundIndex(morton_code octant) const {
        size_t lower_bound = 0;
        size_t upper_bound = this->size();

        // chatgpt magic
        Kokkos::parallel_reduce(
            "LowerBoundSearch", this->size(),
            KOKKOS_LAMBDA(const size_t i, size_t& update) {
                if (octants(i) >= octant) {
                    update = (update < i) ? update : i;
                }
            },
            Kokkos::Min<size_t>(lower_bound));

        if (lower_bound == this->size() || octants(lower_bound) != octant) {
            throw std::runtime_error("Octant not found in AidList.");
        }
        return lower_bound;
    }

    size_t AidList::getUpperBoundIndexExclusive(morton_code octant) const {
        size_t lower_bound = 0;
        size_t upper_bound = this->size();

        Kokkos::parallel_reduce(
            "UpperBoundExclusiveSearch", this->size(),
            KOKKOS_LAMBDA(const size_t i, size_t& update) {
                if (octants(i) > octant) {
                    update = (update < i) ? update : i;
                }
            },
            Kokkos::Min<size_t>(upper_bound));

        return upper_bound;
    }

    size_t AidList::getUpperBoundIndexInclusive(morton_code octant) const {
        size_t lower_bound = 0;
        size_t upper_bound = this->size();

        Kokkos::parallel_reduce(
            "UpperBoundInclusiveSearch", this->size(),
            KOKKOS_LAMBDA(const size_t i, size_t& update) {
                if (octants(i) >= octant) {
                    update = (update < i) ? update : i;
                }
            },
            Kokkos::Min<size_t>(upper_bound));

        if (upper_bound == this->size() || octants(upper_bound) != octant) {
            throw std::runtime_error("Octant not found in AidList.");
        }
        return upper_bound;
    }

    size_t AidList::getNumParticlesInOctant(morton_code octant) const {
        const size_t lower_bound_idx = getLowerBoundIndex(octant);
        const size_t upper_bound_idx = getUpperBoundIndex(octant);

        if (lower_bound_idx > upper_bound_idx) {
            throw std::runtime_error("loweridx > upper_idx in getNumParticlesInOctant...");
        }

        return upper_bound_idx - lower_bound_idx;
    }

}  // namespace ippl