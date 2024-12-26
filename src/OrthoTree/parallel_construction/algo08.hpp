#include "../OrthoTree.h"

/*
TODO:
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::linearise_octants(
        const Kokkos::View<morton_code*>& input_view) {
        /**
         * Idea:
         * 1. we count how many elements we have to copy into our linearised view
         *      -> O(1), so the double calls (probably) dont matter
         * 2. we copy the needed elements into the lienarised_view
         *
         * This way we can initialise our view to spec directly, without having to resize. I dont
         * know if this holds true if input_view contains hunderts of millions of octants, maybe it
         * is more efficient to resize the output then?
         */

        const size_t input_size = input_view.size();

        assert(input_size > 0
               && "Octants.size() is zero, dont call this function with an empty list!");

        // to remove warnings due to KOKKOS_LAMBDA
        const auto local_morton_helper = this->morton_helper;
        size_t count                   = 0;
        Kokkos::parallel_reduce(
            "CountValidOctants", input_size - 1,
            KOKKOS_LAMBDA(const size_t i, size_t& local_count) {
                // no branching this way
                local_count += static_cast<size_t>(
                    !local_morton_helper.is_ancestor(input_view(i + 1), input_view(i)));
            },
            count);

        // last element will always be inserted, hence the 'count + 1'
        const size_t output_size = count + 1;
        Kokkos::View<morton_code*> output_view("linearised_view", output_size);

        Kokkos::View<size_t> index("index");
        Kokkos::deep_copy(index, size_t(0));

        Kokkos::parallel_for(
            "AddRelevantOctants", input_size - 1, KOKKOS_LAMBDA(const size_t i) {
                if (!local_morton_helper.is_ancestor(input_view(i + 1), input_view(i))) {
                    const size_t cur_index = Kokkos::atomic_fetch_add(&index(), size_t(1));
                    output_view(cur_index) = input_view(i);
                }
            });

        // deep copy didnt compile, so we use this georgeous thing lol
        Kokkos::parallel_for(
            "AddLastElement", 1, KOKKOS_LAMBDA(const int) {
                output_view(output_size - 1) = input_view(input_size - 1);
            });

        // i checked and this is always sorted, idk why though
        return output_view;
    }
}  // namespace ippl