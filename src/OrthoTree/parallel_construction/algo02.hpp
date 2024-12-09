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
    Kokkos::View<morton_code*> OrthoTree<Dim>::complete_region(morton_code code_a,
                                                                morton_code code_b) {
        START_FUNC;
        assert(code_a < code_b);

        morton_code nearest_comm_ancestor =
            morton_helper.get_nearest_common_ancestor(code_a, code_b);
        std::vector<morton_code> stack = morton_helper.get_children(nearest_comm_ancestor);

        size_t estimated_size = 1024; 
        Kokkos::View<morton_code*> min_lin_tree("min_lin_tree", estimated_size);
        size_t idx = 0;

        while (stack.size() > 0) {
            morton_code current_node = stack.back();
            stack.pop_back();

            bool is_between_a_b   = (code_a < current_node) && (current_node < code_b);
            bool is_ancestor_of_a = morton_helper.is_ancestor(code_a, current_node);
            bool is_ancestor_of_b = morton_helper.is_ancestor(code_b, current_node);

            if (is_between_a_b && !is_ancestor_of_b) {
                // Resize like vector
                if (idx >= min_lin_tree.extent(0)) {
                    Kokkos::resize(min_lin_tree, 2 * min_lin_tree.extent(0));
                }
                min_lin_tree(idx) = current_node;
                ++idx;
            } else if (is_ancestor_of_a || is_ancestor_of_b) {
                for (morton_code& child : morton_helper.get_children(current_node)) {
                    stack.push_back(child);
                }
            }
        }

        Kokkos::resize(min_lin_tree, idx + 1);  // Final size
        Kokkos::parallel_sort(min_lin_tree);

        END_FUNC;
        return min_lin_tree;
    }
}  // namespace ippl