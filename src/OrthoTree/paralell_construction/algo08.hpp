#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
                                        Kokkos::View<morton_code*> octants);

}  // namespace ippl
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::linearise_octants(
        const Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        logger << "size: " << octants.size() << endl;
        Kokkos::vector<morton_code> linearised;

        for (size_t i = 0; i < octants.size() - 1; ++i) {
            if (morton_helper.is_ancestor(octants[i + 1], octants[i])) {
                continue;
            }

            linearised.push_back(octants[i]);
        }

        linearised.push_back(octants.back());

        logger << "finished, size is: " << linearised.size() << endl;
        END_FUNC;
        return linearised;
    }
}  // namespace ippl