#include "../OrthoTree.h"

/*
TODO:
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
        const Kokkos::View<morton_code*>& octants) {

        IpplTimings::TimerRef lineariseOctantsTimer = IpplTimings::getTimer("linearise_octants");
        IpplTimings::startTimer(lineariseOctantsTimer);


        Kokkos::View<morton_code*> linearised("linearised", octants.size());

        if (octants.size() == 0) {
            IpplTimings::stopTimer(lineariseOctantsTimer);
            return linearised;
        }

        size_t j = 0;
        for (size_t i = 0; i < octants.size() - 1; ++i) {
            if (morton_helper.is_ancestor(octants[i + 1], octants[i])) {
                continue;
            }

            linearised[j] = octants[i];
            ++j;
        }

        linearised[j] = octants[octants.size() - 1];
        Kokkos::resize(linearised, j + 1);

        IpplTimings::stopTimer(lineariseOctantsTimer);
        
        return linearised;
    }
}  // namespace ippl
