#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::block_partition(morton_code min_octant,
                                                               morton_code max_octant) {
        Kokkos::View<morton_code*> T = complete_region(min_octant, max_octant);

        // find the lowest level (smallest depth)
        size_t lowest_level;
        Kokkos::parallel_reduce(
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& min_depth) {
                size_t depth = morton_helper.get_depth(T(i));
                if (depth < min_depth) {
                    min_depth = depth;
                }
            },
            Kokkos::Min<size_t>(lowest_level));

        // count the number of elements at the lowest level
        size_t C_size;
        Kokkos::parallel_reduce(
            T.size(),
            KOKKOS_LAMBDA(const size_t i, size_t& count) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    count++;
                }
            },
            C_size);

        Kokkos::View<morton_code*> C("C_view", C_size);

        // populate C_view
        Kokkos::parallel_scan(
            T.size(), KOKKOS_LAMBDA(const size_t i, size_t& index, bool final) {
                if (morton_helper.get_depth(T(i)) == lowest_level) {
                    if (final) {
                        C(index) = T(i);
                    }
                    index++;
                }
            });

        Kokkos::View<morton_code*> G = complete_tree(C);

        Kokkos::View<size_t*> weights      = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        Kokkos::View<morton_code*> octants = partition(G, weights);

        morton_code new_min_octant = octants[0];
        morton_code new_max_octant = *(octants.data() + octants.extent(0) - 1);
        this->aid_list_m.innitFromOctants(new_min_octant, new_max_octant);

        return octants;
    }
}  // namespace ippl