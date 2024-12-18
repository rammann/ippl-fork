#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_tree(Kokkos::View<morton_code*>& octants);
}  // namespace ippl
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::complete_tree(
        Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        IpplTimings::TimerRef completeTreeTimer = IpplTimings::getTimer("complete_tree");
        IpplTimings::startTimer(completeTreeTimer);
        world_rank = Comm->rank();
        world_size = Comm->size();

        logger << "called with n_octants=" << octants.size() << endl;
        // this removes duplicates, inefficient as of now
        std::map<morton_code, int> m;
        for (auto octant : octants) {
            ++m[octant];
        }

        logger << "map has " << m.size()
            << "octants in total, previously we had: " << octants.size() << endl;

        octants.clear();
        for (const auto [octant, count] : m) {
            octants.push_back(octant);
        }

        octants = linearise_octants(octants);

        Kokkos::vector<size_t> weights(octants.size(), 1);
        octants = partition(octants, weights);

        morton_code first_rank0;
        if (world_rank == 0) {
            const morton_code dfd_root = morton_helper.get_deepest_first_descendant(morton_code(0));
            const morton_code A_finest =
                morton_helper.get_nearest_common_ancestor(dfd_root, octants.front());

            assert(morton_helper.get_depth(A_finest) < max_depth_m);
            const morton_code first_child = morton_helper.get_first_child(A_finest);

            // this imitates push_front
            first_rank0 = first_child;
        }
        else if (world_rank == world_size - 1) {
            const morton_code dld_root = morton_helper.get_deepest_last_descendant(morton_code(0));
            const morton_code A_finest =
                morton_helper.get_nearest_common_ancestor(dld_root, octants.back());
            const morton_code last_child = morton_helper.get_last_child(A_finest);
            assert(morton_helper.get_depth(A_finest) < max_depth_m);
            octants.push_back(last_child);
        }

        if (world_rank > 0) {
            Comm->send(*octants.data(), 1, world_rank - 1, 0);
        }

        morton_code buff;
        if (world_rank < world_size - 1) {
            mpi::Status status;
            Comm->recv(&buff, 1, world_rank + 1, 0, status);
            // do we need a status check here or not?
            octants.push_back(buff);
        }

        Kokkos::vector<morton_code> R;

        // rank 0 works differently, as we need to 'simulate' push_front
        if (world_rank == 0) {
            R.push_back(first_rank0);
            for (morton_code elem : complete_region(first_rank0, octants[0])) {
                R.push_back(elem);
            }
        }

        Comm->barrier();
        if (world_rank == 0) {
            LOG << "GOT HERE" << endl;
        }

        // currently a crash somewhere in this call stack
        const size_t n = octants.size();
        for (size_t i = 0; i < n - 1; ++i) {
            R.push_back(octants[i]);
            for (morton_code elem : complete_region(octants[i], octants[i + 1])) {
                R.push_back(elem);
            }
        }

        Comm->barrier();
        if (world_rank == 0) {
            LOG << "DIDNT GET HERE" << endl;
        }

        if (world_rank == world_size - 1) {
            R.push_back(octants[n - 1]);
        }

        IpplTimings::stopTimer(completeTreeTimer);

        END_FUNC;
        return R;
    }
}  // namespace ippl