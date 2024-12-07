#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include <Kokkos_Pair.hpp>
#include <Kokkos_Vector.hpp>
#include <fstream>
#include <vector>

#include "OrthoTreeTypes.h"

#include "OrthoTreeParticle.h"
#include "helpers/BoundingBox.h"
#include "helpers/Config.h"
#include "helpers/MortonHelper.h"

namespace ippl {

    // this is defined outside of Types.h on purpose, as this is likely to change in the finalised
    // implementation
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
        using particle_t      = particle_type_template<Dim>;
        using bounds_t        = BoundingBox<Dim>;

        using aid_list_t = Kokkos::vector<Kokkos::pair<morton_code, size_t>>;

        const size_t max_depth_m;
        const size_t max_particles_per_node_m;
        const bounds_t root_bounds_m;
        const Morton<Dim> morton_helper;

        // as of now the tree is stored only as morton codes, this needs to be discussed as a group
        Kokkos::vector<morton_code> tree_m;

        // NEW TREE TYPE!
        Kokkos::View<morton_code*> finished_tree;

        size_t n_particles;

        // this list should not be edited, we will probably need it if we implement the tree update
        // as well
        aid_list_t aid_list;

        int world_rank;
        int world_size;

        Inform logger;

    public:
        OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_t& root_bounds);

        /**
         * @brief This is the most basic way to build a tree. Its inefficien, but it (should) be
         * correct. Can be used to compare against parallel implementations later on.
         *
         * @param particles An 'object of arrays' of particles (google it)
         */
        Kokkos::View<morton_code*> build_tree_naive(particle_t const& particles);

#pragma region paralell construction
        /**
         * ALGO 1
         *
         * @brief We take this as an entry function.
         * Each proc will return its own part of the tree?
         *
         * @param particles
         * @return Kokkos::vector<morton_code>
         */
        Kokkos::View<morton_code*> build_tree(particle_t const& particles);

        /**
         * ALGO 2
         *
         * @brief sequential construction of a minimal linear octree between two octants
         *
         * @param morton codes code_a and code_b of the octants
         *
         * @return list of morton codes of minimal linear octree between the two octants
         **/
        Kokkos::vector<morton_code> complete_region(morton_code code_a, morton_code code_b);

        /**
         * ALGO 3
         *
         * @brief Implements the logic part of algorithm 3.
         */
        Kokkos::vector<morton_code> complete_tree(Kokkos::vector<morton_code>& tree);

        /**
         * ALGO 4
         *
         * @brief algorithm 4 parallel partitioning of octants into large contiguous blocks
         *
         * @param unpartitioned octree unpartitioned_tree
         *
         * @return block partitioned octree, and unpartitioned_tree is re-distributed
         **/
        Kokkos::vector<morton_code> block_partition(morton_code min_octant, morton_code max_octant);

        /**
         * ALGO 5
         *
         * @brief This function partitions the workload of building the tree across
         * the available mpi ranks.
         */
        Kokkos::vector<morton_code> partition(Kokkos::vector<morton_code>& octants,
                                              Kokkos::vector<size_t>& weights);

        /**
         * ALGO 8
         *
         * @brief Linearises octants by removing ancestors that would cause overlaps
         * @param list of octants - sorted
         * @return list of linearised octants - sorted
         * @warning THIS FUNCTION ASSUMES THAT THE OCTANTS ARE SORTED
         */
        Kokkos::vector<morton_code> linearise_octants(Kokkos::vector<morton_code> const& octants);
#pragma endregion  // paralell construction

#pragma region balancing

        Kokkos::View<morton_code*> algo6(morton_code octant_N, morton_code descendant_L);

        Kokkos::View<morton_code*> algo7(morton_code octant_N,
                                         Kokkos::View<morton_code*> partial_descendants_L);

        Kokkos::View<morton_code*> algo9(Kokkos::View<morton_code*> sorted_incomplete_tree_L);

        Kokkos::View<morton_code*> algo10(morton_code octant_N,
                                          Kokkos::View<morton_code*> partial_descendants_L);

        Kokkos::View<morton_code*> algo11(Kokkos::View<morton_code*> distributed_complete_tree_L);

#pragma endregion  // balancing

#pragma region helpers
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

        /**
         * @brief Returns a vector with morton_codes and lists of particle ids inside this region
         * This is also not meant to be permanent, but it should suffice for the beginning
         * @return Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>>
         */
        Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>> get_tree() const;

        size_t getAidList_lowerBound(morton_code octant);
        size_t getAidList_upperBound(morton_code octant);

#pragma endregion  // helpers

        /*
         * @brief Returns the number of particles in the octants, asks rank 0 for the number of
         * particles in the octants
         *
         * @param octant vector
         * @return size_t vector - number of particles in the octants
         */
        Kokkos::vector<size_t> get_num_particles_in_octants_parallel(
            const Kokkos::vector<morton_code>& octants);

        // setter for aid list also adapts n_particles
        void set_aid_list(const aid_list_t& aid_list) {
            this->aid_list = aid_list;
            n_particles    = aid_list.size();
        }

        Kokkos::View<morton_code*> build_tree_from_octants(
            const Kokkos::vector<morton_code>& octants);

        void init_aid_list_from_octants(morton_code min_octant, morton_code max_octant);

        std::pair<morton_code, morton_code> get_relevant_aid_list(particle_t const& particles);

    private:
        /**
         * @brief initializes the aid list which has form vec{pair{morton_code, size_t}}
         * will sort the aidlist in ascending morton codes
         *
         * @param particles
         */
        aid_list_t initialize_aid_list(particle_t const& particles);

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
         * @brief counts the number of particles covered by the cell decribed by the morton codes
         *        initialize_aid_list needs to be called first
         *
         * @param vector of morton codes
         * @return number of particles in the cells specified by the morton code vector
         * @warning THIS FUNCTION ASSUMES THAT THE OCTANTS ARE SORTED
         * @TODO parallelize this with Kokkos
         **/
        Kokkos::vector<size_t> get_num_particles_in_octants_seqential(
            const Kokkos::vector<morton_code>& octants);

#pragma region print_helpers

        std::ostream& print_octant(std::ostream& os, morton_code octant) {
            const grid_coordinate grid = morton_helper.decode(octant);
            const auto bounds          = bounds_t::bounds_from_grid_coord(
                root_bounds_m, grid, morton_helper.get_depth(octant), max_depth_m);

            return os << octant << " " << bounds.get_min() << " " << bounds.get_max();
        }

        template<typename Iterator>
        std::ostream& print_octant_list(std::ostream& os, Iterator begin, Iterator end) {
            for (Iterator it = begin; it != end; ++it) {
                print_octant(os, *it);
                os << std::endl;
            }

            return os;
        }

        std::ostream& print_particles(std::ostream& os, particle_t const& particles) {
            for (size_t i = 0; i < particles.getLocalNum(); ++i) {
                os << i << " " << particles.R(i) << std::endl;
            }

            return os;
        }

        void particles_to_file(particle_t const& particles) {
            std::string outputPath = std::string(IPPL_SOURCE_DIR)
                                     + "/src/OrthoTree/scripts/output/particles"
                                     + std::to_string(Comm->rank()) + ".txt";

            std::ofstream file(outputPath, std::ofstream::out);
            print_particles(file, particles);
            file.flush();
            file.close();
        }

        /**
         * @brief prints the octants to a file
         *
         * @param octants
         * @template T a container of morton codes that supports data() and size()
         */
        template<typename T>
        void octants_to_file(const T& octants) {
            std::string outputPath = std::string(IPPL_SOURCE_DIR)
                                     + "/src/OrthoTree/scripts/output/octants"
                                     + std::to_string(Comm->rank()) + ".txt";

            std::ofstream file(outputPath, std::ofstream::out);
            print_octant_list(file, octants.data(), octants.data() + octants.size());
            file.flush();
            file.close();
        }

    };
#pragma endregion  // print_helpers

}  // namespace ippl

#include "OrthoTree.hpp"

// implementations of construction algos
#include "paralell_construction/algo01.hpp"
#include "paralell_construction/algo02.hpp"
#include "paralell_construction/algo03.hpp"
#include "paralell_construction/algo04.hpp"
#include "paralell_construction/algo05.hpp"
#include "paralell_construction/algo08.hpp"

// implementations of balancing algos
// #include "balancing/algo06.hpp"
// #include "balancing/algo07.hpp"
// #include "balancing/algo09.hpp"
// #include "balancing/algo10.hpp"
// #include "balancing/algo11.hpp"

#endif // ORTHOTREE_GUARD
