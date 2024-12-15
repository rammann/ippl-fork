#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    auto OrthoTree<Dim>::block_partition(morton_code min_octant, morton_code max_octant) {
        auto T = complete_region(min_octant, max_octant);

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

        auto G = complete_tree(C);

        auto weights = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        auto octants = partition(G, weights);

        this->aid_list_m.innitFromOctants(*octants.data(), *(octants.data() + octants.size() - 1));

        return octants;
    }
}  // namespace ippl