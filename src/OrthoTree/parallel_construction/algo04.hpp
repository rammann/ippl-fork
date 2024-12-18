#include "../OrthoTree.h"

/*
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    auto OrthoTree<Dim>::block_partition(morton_code min_octant, morton_code max_octant) {
        START_FUNC;
        IpplTimings::TimerRef timer = IpplTimings::getTimer("block_partition");
        IpplTimings::clearTimer(timer);
        IpplTimings::startTimer(timer);

        auto T = complete_region(min_octant, max_octant);

        // the lowest level is actually the 'highest' (closest to root) node in our tree
        size_t lowest_level = morton_helper.get_depth(*std::min_element(
            T.data(), T.data() + T.size(), [this](const morton_code& a, const morton_code& b) {
                return morton_helper.get_depth(a) < morton_helper.get_depth(b);
            }));

        // we only use the 'highest' octants
        Kokkos::vector<morton_code> C;
        for (auto it = T.data(); it != T.data() + T.size(); ++it) {
            const morton_code octant = *it;
            if (morton_helper.get_depth(octant) == lowest_level) {
                C.push_back(octant);
            }
        }

        auto G = complete_tree(C);

        auto weights = this->aid_list_m.getNumParticlesInOctantsParallel(G);
        auto octants = partition(G, weights);

        IpplTimings::TimerRef innitfromoctants = IpplTimings::getTimer("innitFromOctants");
        IpplTimings::clearTimer(innitfromoctants);
        IpplTimings::startTimer(innitfromoctants);

        this->aid_list_m.innitFromOctants(*octants.data(), *(octants.data() + octants.size() - 1));

        IpplTimings::stopTimer(innitfromoctants);
        
        IpplTimings::stopTimer(timer);
        END_FUNC;
        return octants;
    }
}  // namespace ippl