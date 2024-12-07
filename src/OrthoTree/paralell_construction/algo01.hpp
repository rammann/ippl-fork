#include "../OrthoTree.h"
/*
TODO:
- WRITE TESTS FOR THE FUNCTION
*/

namespace ippl {
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree(particle_t const& particles) {
        START_FUNC;

        world_size = Comm->size();
        world_rank = Comm->rank();  // TODO: move this to constructor, but then all tests need a
                                    // main to init ippl

        this->aid_list_m.initialize(root_bounds_m, particles);
        auto [min_octant, max_octant] = this->aid_list_m.getMinReqOctants();

        auto octants = block_partition(min_octant, max_octant);

        particles_to_file(particles);

        // Each proc has now as much of the aid_list as he needs and can start building the
        // tree.
        Kokkos::View<morton_code*> tree_view = build_tree_from_octants(octants);
        octants_to_file(tree_view);
        Comm->barrier();
        logger.flush();

        size_t total_particles = 0;
        for (size_t i = 0; i < tree_view.size(); ++i) {
            auto num_particles = this->aid_list_m.getNumParticlesInOctant(tree_view[i]);
            total_particles += num_particles;
        }

        size_t global_total_particles = 0;
        Comm->allreduce(&total_particles, &global_total_particles, 1, std::plus<size_t>());

        Comm->barrier();

        {
            const size_t col_width = 15;

            auto printer = [col_width](const auto& rank, const auto& tree_size,
                                       const auto& num_octants_from_algo_4,
                                       const auto& num_particles) {
                std::cerr << std::left << std::setw(col_width) << rank << std::left
                          << std::setw(col_width) << tree_size << std::left << std::setw(col_width)
                          << num_octants_from_algo_4 << std::left << std::setw(col_width)
                          << num_particles << std::endl;
            };

            {
                int ring_buf;
                if (world_rank + 1 < world_size) {
                    mpi::Status status;
                    Comm->recv(&ring_buf, 1, world_rank + 1, 0, status);
                } else {
                    std::cerr << std::string(col_width * 3, '=') << std::endl;
                    printer("rank", "octs_now", "octs_4", "particles");
                    std::cerr << std::string(col_width * 3, '-') << std::endl;
                }

                printer(world_rank, tree_view.size(), octants.size(), total_particles);

                if (world_rank > 0) {
                    Comm->send(ring_buf, 1, world_rank - 1, 0);
                }
            }

            if (world_rank == 0) {
                std::cerr << std::string(col_width * 3, '-') << std::endl
                          << "We now have " << global_total_particles << " particles" << std::endl
                          << "("
                          << (100.0 * (double)global_total_particles
                              / (double)particles.getTotalNum())
                          << "% of starting value lol)" << std::endl
                          << std::string(col_width * 3, '-') << std::endl;
            }
        }

        //these following lines are for testing purposes
        //they check that the resulting tree is sorted and doesn't leave gaps 
        //in the domain
        // The lambda checks that two morton codes are sorted and that there are 
        // no other morton codes between them that overlaps neither
        auto is_sorted_and_contiguous = [this](const morton_code& a, const morton_code& b) {
            if (b <= a || morton_helper.is_descendant(b, a)) {
                return false;
            }
            return b <= morton_helper.get_deepest_first_descendant(a + morton_helper.get_step_size(a));
        };
        assert(std::is_sorted(tree_view.data(), tree_view.data() + tree_view.size(), is_sorted_and_contiguous)
               && "partitioned_tree is not sorted");
        return tree_view;
    }

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree_from_octants(
        const Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        /**
         * THIS CAN PROBABLY BE DONE MUCH SMARTER AND FASTER!!
         */

        std::stack<std::pair<morton_code, size_t>> s;

        for (size_t i = 0; i < octants.size(); ++i) {
            s.push({octants[i], this->aid_list_m.getNumParticlesInOctant(octants[i])});
        }

        std::vector<morton_code> result_tree;  // TODO: remove this!
        while (!s.empty()) {
            const auto& [octant, count] = s.top();
            s.pop();

            if (count <= max_particles_per_node_m
                || morton_helper.get_depth(octant) >= max_depth_m) {
                result_tree.push_back(octant);
                continue;
            }

            for (const auto& child_octant : morton_helper.get_children(octant)) {
                const size_t count = this->aid_list_m.getNumParticlesInOctant(child_octant);

                s.push({child_octant, count});
            }
        }

        // if we sort the tree after construction we can compare two trees
        std::sort(result_tree.begin(), result_tree.end());
        Kokkos::View<morton_code*> return_tree("result_tree", result_tree.size());
        for (size_t i = 0; i < result_tree.size(); ++i) {
            return_tree(i) = result_tree[i];
        }
        END_FUNC;
        return return_tree;
    }
}  // namespace ippl