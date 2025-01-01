#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
        morton_code max_octant) {
            
        IpplTimings::TimerRef blockPartitionTimer = IpplTimings::getTimer("block_partition");
        IpplTimings::startTimer(blockPartitionTimer);

        Kokkos::View<morton_code*> T = complete_region(min_octant, max_octant);

        // the lowest level is actually the 'highest' (closest to root) node in our tree
        size_t lowest_level = morton_helper.get_depth(*std::min_element(
            T.data(), T.data() + T.size(), [this](const morton_code& a, const morton_code& b) {
                return morton_helper.get_depth(a) < morton_helper.get_depth(b);
            }));

        const size_t C_size =
            std::accumulate(T.data(), T.data() + T.size(), 0,
                [this, lowest_level](auto acc, const morton_code octant) {
                    return acc + (morton_helper.get_depth(octant) == lowest_level);
                });

        // we only use the 'highest' octants
        Kokkos::View<morton_code*> C("C_view", C_size);
        size_t C_index = 0;
        for (auto it = T.data(); it != T.data() + T.size(); ++it) {
            const morton_code octant = *it;
            if (morton_helper.get_depth(octant) == lowest_level) {
                C[C_index] = octant;
                ++C_index;
            }
        }

        Kokkos::View<morton_code*> G = complete_tree(C);

        Kokkos::View<size_t*> weights = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        Kokkos::View<morton_code*> octants = partition(G, weights);

        morton_code min_step = morton_helper.get_step_size(max_depth_m);
        morton_code max_parent = *(octants.data() + octants.size() - 1);

        morton_code new_min_octant = morton_helper.get_deepest_first_descendant(octants[0]);
        morton_code new_max_octant = morton_helper.get_deepest_last_descendant(max_parent) + min_step;

        IpplTimings::TimerRef innitfromoctants = IpplTimings::getTimer("innitfromoctants");
        IpplTimings::startTimer(innitfromoctants);

        this->aid_list_m.innitFromOctants(new_min_octant, new_max_octant);

        IpplTimings::stopTimer(innitfromoctants);

        IpplTimings::stopTimer(blockPartitionTimer);
        return octants;
    }
}  // namespace ippl
