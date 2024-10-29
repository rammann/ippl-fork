#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include "Types.h"
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


        // probably best kept as member, need to compare with new particle codes for update
        aid_list_t aid_list;

    public:

        OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_t& root_bounds)
            : max_depth_m(max_depth), max_particles_per_node_m(max_particles_per_node),
            root_bounds_m(root_bounds), morton_helper(max_depth)
        { }
        /**
         * @brief algorithm 1' topdown sequential construction of octree
         *
         * @param morton code of root_node, particles
         *
         * @return list of morton codes of leaves of tree spanning root node
         */

        ippl::vector_t<morton_code> build_tree_topdown_sequential(morton_code root_node, particle_t const& particles)
        {
            // insert the root into the tree
          ippl::vector_t<morton_code> tree; // subtree that will be returned
          ippl::vector_t<morton_code> stack; // stack used to build the tree


            stack.push_back(root_node); // maybe store morton_code(0) s.t. we can call morton::root_val or smth

            initialize_aid_list(particles); // initialize aid list - required for particle counting could be moved to constructor

            while(stack.size() > 0){
                morton_code current_node = stack.back();
                stack.pop_back();

              if(NumberOfParticles(current_node) > max_particles_per_node_m && morton_helper.get_depth(current_node) < max_depth_m){
                
                ippl::vector_t<morton_code> children = morton_helper.get_children(current_node);
                
                for(morton_code child : children){stack.push_back(child);}

              }
              else{
                tree.push_back(current_node);
              }
            }

            return tree;
        }

        /**
         * @brief returns a list of morton_codes (nodes) / particle ids
         * this is intended for testing
         */
        //Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>> get_orthotree() const;

        // todo @aaron, generic tree walker that takes 2 funcs (select & apply)
        void traverse_tree();

    private:

        /**
         * @brief initializes the aid list which has form vec{pair{morton_code, size_t}}
         * will sort the aidlist in ascending morton codes
         *
         * @param particles
         */
        void initialize_aid_list(particle_t const& particles)
        {
            // maybe get getGlobalNum()?
            const size_t n_particles = particles.getLocalNum();
            const size_t nodes_per_edge = (size_t(1) << max_depth_m);

            // store dimensions of root bounding box
            const real_coordinate root_bounds_size = root_bounds_m.get_max() - root_bounds_m.get_min();

            aid_list.resize(n_particles);

            for ( size_t i = 0; i < n_particles; ++i ) {
                // real/grid_coordinate has lazy eval
                const real_coordinate normalized = (particles.R(i) - root_bounds_m.get_min()) / root_bounds_size;
                const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * nodes_per_edge);
                aid_list[i] = { morton_helper.encode(grid_coord, max_depth_m), i };
            }

            std::sort(aid_list.begin(), aid_list.end(), [ ] (const auto& a, const auto& b)
            {
                return a.first < b.first;
            });
        }
        /**
         * @brief counts the number of particles covered by the cell decribed by the morton code
         *        initialize_aid_list needs to be called first
         *
         * @params morton_code
         *
         * @return number of particles in the cell specified by the morton code
         *
         **/

        size_t NumberOfParticles(morton_code mc){
            size_t count = 0;
            for(auto aid_list_el : aid_list){
              count += morton_helper.is_ancestor(aid_list_el.first,mc); // iff cell is ancestor of particle code of the corresponding particle, the particle is in the box 
            }

            return count;
        }
          
    };

} // namespace ippl

#endif // ORTHOTREE_GUARD
