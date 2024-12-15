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
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
        const Kokkos::View<morton_code*>& octants) {
        START_FUNC;
        logger << "size: " << octants.size() << endl;
        Kokkos::View<morton_code*> linearised("linearised", octants.size());

        size_t j = 0;
        for (size_t i = 0; i < octants.size() - 1; ++i) {
            if (morton_helper.is_ancestor(octants[i + 1], octants[i])) {
                continue;
            }

            linearised[j] = octants[i];
            ++j;
        }

        linearised[j] = octants[octants.size() - 1];
        Kokkos::resize(linearised, j+1);

        logger << "finished, size is: " << linearised.size() << endl;
        END_FUNC;
        return linearised;
    }
}  // namespace ippl