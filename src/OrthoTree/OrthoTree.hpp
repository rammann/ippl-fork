#include <ostream>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "Communicate/Operations.h"
#include "Communicate/Request.h"
#include "Kokkos_Vector.hpp"
#include "OrthoTree.h"

namespace ippl {

    template <size_t Dim>
    OrthoTree<Dim>::OrthoTree(size_t max_depth, size_t max_particles_per_node,
                              const bounds_t& root_bounds)
        : max_depth_m(max_depth)
        , max_particles_per_node_m(max_particles_per_node)
        , root_bounds_m(root_bounds)
        , morton_helper(max_depth)
        , aid_list_m(AidList<Dim>(max_depth))
        , logger("OrthoTree", std::cout, INFORM_ALL_NODES) {
        logger.setOutputLevel(5);
        logger.setPrintNode(INFORM_ALL_NODES);
    }

    inline static int stack_depth = 0;

#define LOG logger << std::string(stack_depth * 2, ' ') << __func__ << ": "

#define END_FUNC               \
    LOG << "FINISHED" << endl; \
    --stack_depth

#define START_FUNC \
    ++stack_depth; \
    LOG << "STARTING" << endl

    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree_naive(particle_t const& particles) {
        // this needs to be initialized before constructing the tree
        this->aid_list_m.initialize(root_bounds_m, particles);

        Kokkos::vector<morton_code> result_tree;

        std::stack<std::pair<morton_code, size_t>> s;
        s.push({ morton_code(0), particles.getLocalNum() });

        while ( !s.empty() ) {
            const auto& [octant, count] = s.top(); s.pop();

            if ( count <= max_particles_per_node_m || morton_helper.get_depth(octant) >= max_depth_m ) {
                result_tree.push_back(octant);
                continue;
            }

            for ( const auto& child_octant : morton_helper.get_children(octant) ) {
                const size_t count = aid_list_m.getNumParticlesInOctant(child_octant);

                // no need to push in this case
                if ( count > 0 ) {
                    s.push({ child_octant, count });
                }
            }
        }

        // if we sort the tree after construction we can compare two trees
        std::sort(result_tree.begin(), result_tree.end());

        Kokkos::View<morton_code*> tree_view("tree_view_naive", result_tree.size());
        tree_view.assign_data(result_tree.data());

        return tree_view;
    }

    template <size_t Dim>
    bool OrthoTree<Dim>::operator==(const OrthoTree& other) {
        if (n_particles != other.n_particles) {
            return false;
        }

        if (tree_m.size() != other.tree_m.size()) {
            return false;
        }

        for (size_t i = 0; i < tree_m.size(); ++i) {
            if (tree_m[i] != other.tree_m[i]) {
                return false;
            }
        }

        return true;
    }
}  // namespace ippl
