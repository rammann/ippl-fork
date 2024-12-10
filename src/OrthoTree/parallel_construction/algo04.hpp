#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant);
}  // namespace ippl
*/

namespace ippl {
    template <size_t Dim>
    auto OrthoTree<Dim>::block_partition(morton_code min_octant, morton_code max_octant) {
        // using auto here, as it returns a Kokkos::vector rn, but will return a Kokkos::View one
        // day
        auto T = complete_region(min_octant, max_octant);

        size_t lowest_level = morton_helper.get_depth(*std::min_element(
            T.data(), T.data() + T.size(), [this](const morton_code& a, const morton_code& b) {
                return morton_helper.get_depth(a) < morton_helper.get_depth(b);
            }));

        // TODO: CHANGE THIS TO KOKKOS::VIEW
        Kokkos::vector<morton_code> C;
        for (morton_code octant : T) {
            if (morton_helper.get_depth(octant) == lowest_level) {
                C.push_back(octant);
            }
        }

        auto G = complete_tree(C);

        auto weights = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        auto octants = partition(G, weights);

        this->aid_list_m.innitFromOctants(*octants.data(), *(octants.data() + octants.size() - 1));

        return octants;
    }
}  // namespace ippl