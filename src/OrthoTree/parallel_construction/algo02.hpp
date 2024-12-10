#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_region(morton_code code_a,
                                                               morton_code code_b);
}  // namespace ippl
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::complete_region(morton_code code_a,
                                                                morton_code code_b) {
        START_FUNC;
        assert(code_a < code_b);

        morton_code neares_comm_ancestor =
            morton_helper.get_nearest_common_ancestor(code_a, code_b);
        std::vector<morton_code> stack = morton_helper.get_children(neares_comm_ancestor);

        Kokkos::vector<morton_code> min_lin_tree;
        while (stack.size() > 0) {
            morton_code current_node = stack.back();
            stack.pop_back();

            bool is_between_a_b   = (code_a < current_node) && (current_node < code_b);
            bool is_ancestor_of_a = morton_helper.is_ancestor(code_a, current_node);
            bool is_ancestor_of_b = morton_helper.is_ancestor(code_b, current_node);

            if (is_between_a_b && !is_ancestor_of_b) {
                min_lin_tree.push_back(current_node);
            } else if (is_ancestor_of_a || is_ancestor_of_b) {
                for (morton_code& child : morton_helper.get_children(current_node)) {
                    stack.push_back(child);
                }
            }
        }

        std::sort(min_lin_tree.begin(), min_lin_tree.end());

        END_FUNC;
        return min_lin_tree;
    }
}  // namespace ippl