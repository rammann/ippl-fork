#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include "OrthoTreeTypes.h"
#include "OrthoTreeParticle.h"
#include "MortonHelper.h"
#include "BoundingBox.h"

#include <Kokkos_Vector.hpp>
#include <Kokkos_Pair.hpp>

#include <vector>
namespace ippl {

    // this is defined outside of Types.h on purpose, as this is likely to change in the finalised implementation
    template <size_t Dim>
    using particle_type_template = OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>>;

    /**
     * @brief This is the OrthoTree class, nice oder.
     *
     * The idea is:
     *
     * - no complexity
     *      auto tree = OrthoTree<Dimension>(max_depth, max_particles, root_bounds);
     *
     * - all the computational complexity
     *      tree.function_that_builds_the_tree(particles);
     *
     * - maybe some complexity, idc
     *      tree.traverse() or whatever operations your heart desieres.
     *
     * - this way we can later implement update as:
     *      tree.update(updated_particles);
     *
     * @tparam Dim
     */
    template <size_t Dim>
    class OrthoTree {
        using real_coordinate = real_coordinate_template<Dim>;
        using grid_coordinate = grid_coordinate_template<Dim>;
        using particle_t = particle_type_template<Dim>;
        using bounds_t = BoundingBox<Dim>;

        using aid_list_t = Kokkos::vector<Kokkos::pair<morton_code, size_t>>;

        const size_t max_depth_m;
        const size_t max_particles_per_node_m;
        const bounds_t root_bounds_m;
        const Morton<Dim> morton_helper;

        // as of now the tree is stored only as morton codes, this needs to be discussed as a group
        Kokkos::vector<morton_code> tree_m;

        size_t n_particles;

        // this list should not be edited, we will probably need it if we implement the tree update as well
        aid_list_t aid_list;

    public:

        OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_t& root_bounds);

        /**
         * @brief This is the most basic way to build a tree. Its inefficien, but it (should) be correct.
         * Can be used to compare against parallel implementations later on.
         *
         * @param particles An 'object of arrays' of particles (google it)
         */
        void build_tree_naive(particle_t const& particles);

        /**
         *
         * @brief This function partitions the workload of building the tree across
         * the available mpi ranks.
         *
         */
        Kokkos::vector<morton_code> partition(Kokkos::vector<morton_code>& octants, Kokkos::vector<size_t>& weights);

        /**
         * @brief Returns a vector with morton_codes and lists of particle ids inside this region
         * This is also not meant to be permanent, but it should suffice for the beginning
         * @return Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>>
         */
        Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>> get_tree() const;

        /**
         * @brief Compares the following aspects of the trees:
         * - n_particles
         * - tree_m.size()
         * - contents of tree_m
         *
         * @warning THIS FUNCTION ASSUMES THAT THE TREES ARE SORTED
         *
         * @param other
         * @return true
         * @return false
         */
        bool operator==(const OrthoTree& other);

    private:

        /**
         * @brief initializes the aid list which has form vec{pair{morton_code, size_t}}
         * will sort the aidlist in ascending morton codes
         *
         * @param particles
         */
        void initialize_aid_list(particle_t const& particles);

        /**
         * @brief counts the number of particles covered by the cell decribed by the morton code
         *        initialize_aid_list needs to be called first
         *
         * @param morton_code
         * @return number of particles in the cell specified by the morton code
         **/
        size_t get_num_particles_in_octant(morton_code octant);

    public:
        // SIMONS FUNCTIONS DONT EDIT, TOUCH OR USE THIS IN YOUR CODE:

        /**
         * @brief algorithm 1' topdown sequential construction of octree
         *
         * @param morton code of root_node, particles
         *
         * @return list of morton codes of leaves of tree spanning root node
         **/
        ippl::vector_t<morton_code> build_tree_topdown_sequential(morton_code root_node, particle_t const& particles)
        {
            // insert the root into the tree
            ippl::vector_t<morton_code> tree;
            ippl::vector_t<morton_code> stack; // stack used to build the tree

            stack.push_back(root_node); // maybe store morton_code(0) s.t. we can call morton::root_val or smth

            initialize_aid_list(particles); // initialize aid list - required for particle counting could be moved to constructor

            while ( stack.size() > 0 ) {
                morton_code current_node = stack.back();
                stack.pop_back();

                if ( get_num_particles_in_octant(current_node) > max_particles_per_node_m && morton_helper.get_depth(current_node) < max_depth_m ) {

                    ippl::vector_t<morton_code> children = morton_helper.get_children(current_node);

                    for ( morton_code child : children ) { stack.push_back(child); }

                }
                else {
                    tree.push_back(current_node);
                }
            }

            std::sort(tree.begin(), tree.end());
            return tree;
        }
    };

} // namespace ippl

#include "OrthoTree.hpp"

#endif // ORTHOTREE_GUARD
