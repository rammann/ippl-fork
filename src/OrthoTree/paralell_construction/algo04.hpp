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
        octants_to_file(T);
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

        logger << "C.size()=" << C.size() << endl;
        Kokkos::vector<morton_code> G = complete_tree(C);
        logger << "we now have n_octants = " << G.size() << endl;

        Kokkos::vector<size_t> weights = get_num_particles_in_octants_parallel(G);
        logger << "weights have size: " << weights.size() << endl;
        /*
        for (size_t i = 0; i < G.size(); ++i) {
            morton_code base_tree_octant = G[i];
            weights[i]                   = std::count_if(
                starting_octants.begin(), starting_octants.end(),
                [&base_tree_octant, this](const morton_code& unpartitioned_tree_octant) {
                    return (unpartitioned_tree_octant == base_tree_octant)
                           || (morton_helper.is_descendant(unpartitioned_tree_octant,
                                                                             base_tree_octant));
                });
        }
        */

        auto partitioned_tree = partition(G, weights);
        // TODO: THIS MIGHT BE WRONG? this is not needed, we sync the aid list outside of this
        /*
        Kokkos::vector<morton_code> global_unpartitioned_tree;
        global_unpartitioned_tree.push_back(min_octant);
        global_unpartitioned_tree.push_back(max_octant);
        starting_octants.clear();
        for (morton_code gup_octant : global_unpartitioned_tree) {
            for (const morton_code& p_octant : partitioned_tree) {
                if (gup_octant == p_octant || morton_helper.is_descendant(gup_octant, p_octant)) {
                    starting_octants.push_back(gup_octant);
                    break;
                }
            }
        }
*/
        logger << "finished, partitioned_tree.size() = " << partitioned_tree.size() << endl;
        END_FUNC;
        return partitioned_tree;
    }
}  // namespace ippl