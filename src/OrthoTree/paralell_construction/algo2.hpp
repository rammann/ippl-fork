#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
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
        morton_code nearest_common_ancestor =
            morton_helper.get_nearest_common_ancestor(code_a, code_b);
        ippl::vector_t<morton_code> stack = morton_helper.get_children(nearest_common_ancestor);
        Kokkos::vector<morton_code> min_lin_tree;

        while (stack.size() > 0) {
            morton_code current_node = stack.back();
            stack.pop_back();

            if ((code_a < current_node) && (current_node < code_b)
                && !morton_helper.is_ancestor(code_b, current_node)) {
                min_lin_tree.push_back(current_node);
            } else if (morton_helper.is_ancestor(code_a, current_node)
                       || morton_helper.is_ancestor(code_b, current_node)) {
                for (morton_code& child : morton_helper.get_children(current_node))
                    stack.push_back(child);
            }
        }

        std::sort(min_lin_tree.begin(), min_lin_tree.end());

        logger << "finished complete_region" << endl;
        END_FUNC;
        return min_lin_tree;
    }
}  // namespace ippl