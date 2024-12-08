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
    Kokkos::vector<morton_code> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                                morton_code max_octant) {
        START_FUNC;
        logger << "called with min: " << min_octant << ", max: " << max_octant << endl;

        Kokkos::vector<morton_code> T = complete_region(min_octant, max_octant);

        logger << "T.size() = " << T.size() << endl;

        Kokkos::vector<morton_code> C;
        size_t lowest_level = std::numeric_limits<morton_code>::max();
        for (const morton_code& octant : T) {
            lowest_level = std::min(lowest_level, morton_helper.get_depth(octant));
        }

        for (morton_code octant : T) {
            if (morton_helper.get_depth(octant) == lowest_level) {
                C.push_back(octant);
            }
        }

        Kokkos::vector<morton_code> G = complete_tree(C);

        Kokkos::vector<size_t> weights = this->aid_list_m.getNumParticlesInOctantsParalell(G);

        auto octants = partition(G, weights);

        this->aid_list_m.innitFromOctants(octants.front(), octants.back());

        END_FUNC;
        return octants;
    }
}  // namespace ippl