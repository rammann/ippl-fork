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

    morton_code AidList::getOctant(size_t idx) const {
        if (idx >= aid_list_m.extent(0)) {
            throw std::runtime_error(
                "On RANK: " + std::to_string(world_rank) + " index: " + std::to_string(idx)
                + " is out of range for AidList of size: " + std::to_string(aid_list_m.extent(0)));
        }

        return aid_list_m(idx, 0);
    }

    morton_code AidList::getID(size_t idx) const {
        if (idx >= aid_list_m.extent(0)) {
            throw std::runtime_error(
                "On RANK: " + std::to_string(world_rank) + " index: " + std::to_string(idx)
                + " is out of range for AidList of size: " + std::to_string(aid_list_m.extent(0)));
        }

        return aid_list_m(idx, 1);
    }

    size_t AidList::size() const {
        return aid_list_m.extent(0);
    }

    std::pair<morton_code, morton_code> AidList::getMinReqOctants() {
        Kokkos::View<morton_code[2]> min_max_octants("min_max_octants");

        if (world_rank == 0) {
            const size_t size        = aid_list_m.extent(0);
            const size_t batch_size  = size / world_size;
            const size_t rank_0_size = batch_size + (size % batch_size);

            for (size_t target_rank = 1; target_rank < world_size; ++target_rank) {
                const size_t start = ((target_rank - 1) * batch_size) + rank_0_size;
                const size_t end   = start + batch_size - 1;

                // Assign min and max to the view
                min_max_octants(0) = getOctant(start);
                min_max_octants(1) = getOctant(end);

                // Send min and max directly from the view
                Comm->send(*min_max_octants.data(), 2, target_rank, 0);
            }

            // Assign min/max for rank 0
            min_max_octants(0) = aid_list_m(0, 0);
            min_max_octants(1) = aid_list_m(rank_0_size - 1, 0);
        } else {
            mpi::Status status;
            Comm->recv(min_max_octants.data(), 2, 0, 0, status);
        }

        auto host_min_max = Kokkos::create_mirror_view(min_max_octants);
        Kokkos::deep_copy(host_min_max, min_max_octants);

        return std::make_pair(host_min_max(0), host_min_max(1));
    }

    size_t AidList::getLowerBoundIndex(morton_code octant) const {
        // Create a host mirror of the aid_list_m view
        auto host_aid_list = Kokkos::create_mirror_view(aid_list_m);
        Kokkos::deep_copy(host_aid_list, aid_list_m);

        // Find the first occurrence of the octant
        for (size_t i = 0; i < aid_list_m.extent(0); ++i) {
            if (host_aid_list(i, 0) >= octant) {
                if (host_aid_list(i, 0) == octant) {
                    return i;
                }
                break;
            }
        }

        throw std::runtime_error("Octant not found in AidList.");
    }

    size_t AidList::getUpperBoundIndex(morton_code octant) const {
        // Create a host mirror of the aid_list_m view
        auto host_aid_list = Kokkos::create_mirror_view(aid_list_m);
        Kokkos::deep_copy(host_aid_list, aid_list_m);

        // Find the position after the last occurrence of the octant
        for (size_t i = 0; i < aid_list_m.extent(0); ++i) {
            if (host_aid_list(i, 0) > octant) {
                return i;
            }
        }

        return aid_list_m.extent(0);
    }

}  // namespace ippl