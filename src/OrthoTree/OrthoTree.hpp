#include "OrthoTree.h"

namespace ippl {

    template <size_t Dim>
    OrthoTree<Dim>::OrthoTree(size_t max_depth, size_t max_particles_per_node, const bounds_t& root_bounds)
        : max_depth_m(max_depth), max_particles_per_node_m(max_particles_per_node),
        root_bounds_m(root_bounds), morton_helper(max_depth)
    { }

    template <size_t Dim>
    void OrthoTree<Dim>::build_tree_naive(particle_t const& particles)
    {
        // this needs to be initialized before constructing the tree
        initialize_aid_list(particles);

        std::stack<std::pair<morton_code, size_t>> s;
        s.push({ morton_code(0), particles.getLocalNum() });

        while ( !s.empty() ) {
            const auto& [octant, count] = s.top(); s.pop();

            if ( count <= max_particles_per_node_m || morton_helper.get_depth(octant) >= max_depth_m ) {
                tree_m.push_back(octant);
                continue;
            }

            for ( const auto& child_octant : morton_helper.get_children(octant) ) {
                const size_t count = get_num_particles_in_octant(child_octant);

                // no need to push in this case
                if ( count > 0 ) {
                    s.push({ child_octant, count });
                }
            }
        }

        // if we sort the tree after construction we can compare two trees
        std::sort(tree_m.begin(), tree_m.end());
    }

    template <size_t Dim>
    bool OrthoTree<Dim>::operator==(const OrthoTree& other)
    {
        if ( n_particles != other.n_particles ) {
            return false;
        }

        if ( tree_m.size() != other.tree_m.size() ) {
            return false;
        }

        for ( size_t i = 0; i < tree_m.size(); ++i ) {
            if ( tree_m[i] != other.tree_m[i] ) {
                return false;
            }
        }

        return true;
    }

    template <size_t Dim>
    Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>> OrthoTree<Dim>::get_tree() const
    {
        Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>> result;
        result.reserve(tree_m.size());
        for ( auto octant : tree_m ) {
            Kokkos::vector<size_t> particle_ids;
            for ( const auto& [particle_code, id] : aid_list ) {
                if ( morton_helper.is_descendant(particle_code, octant) ) {
                    particle_ids.push_back(id);
                }
            }

            result.push_back(Kokkos::make_pair(octant, particle_ids));
        }

        return result;
    }

    template <size_t Dim>
    void OrthoTree<Dim>::initialize_aid_list(particle_t const& particles)
    {
        // maybe get getGlobalNum() in the future?
        n_particles = particles.getLocalNum();
        const size_t grid_size = (size_t(1) << max_depth_m);

        // store dimensions of root bounding box
        const real_coordinate root_bounds_size = root_bounds_m.get_max() - root_bounds_m.get_min();

        aid_list.resize(n_particles);

        for ( size_t i = 0; i < n_particles; ++i ) {
            // normalize particle coordinate inside the grid
            // particle locations are accessed with .R(index)
            const real_coordinate normalized = (particles.R(i) - root_bounds_m.get_min()) / root_bounds_size;

            // calculate the grid coordinate relative to the bounding box and grid size
            const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * grid_size);
            aid_list[i] = { morton_helper.encode(grid_coord, max_depth_m), i };
        }

        // list is sorted by asccending morton codes
        std::sort(aid_list.begin(), aid_list.end(), [ ] (const auto& a, const auto& b)
        {
            return a.first < b.first;
        });
    }

    template <size_t Dim>
    size_t OrthoTree<Dim>::get_num_particles_in_octant(morton_code octant)
    {
        const morton_code lower_bound_target = octant;
        // this is the same logic as in Morton::is_ancestor/Morton::is_descendant
        const morton_code upper_bound_target = octant + morton_helper.get_step_size(octant);

        auto lower_bound_idx = std::lower_bound(aid_list.begin(), aid_list.end(), lower_bound_target,
        [ ] (const Kokkos::pair<unsigned long long, unsigned long>& pair, const morton_code& val)
        {
            return pair.first < val;
        });

        auto upper_bound_idx = std::upper_bound(aid_list.begin(), aid_list.end(), upper_bound_target,
            [ ] (const morton_code& val, const Kokkos::pair<unsigned long long, unsigned long>& pair)
        {
            return val < pair.first;
        });

        return static_cast<size_t>(upper_bound_idx - lower_bound_idx);
    }

    template <size_t Dim>
    ippl::vector_t<morton_code> OrthoTree<Dim>::complete_region(morton_code code_a, morton_code code_b) 
    { 
      morton_code nearest_common_ancestor = morton_helper.get_nearest_common_ancestor(code_a, code_b);
      ippl::vector_t<morton_code> trial_nodes = morton_helper.get_children(nearest_common_ancestor);
      ippl::vector_t<morton_code> min_lin_tree;

      while (trial_nodes.size() > 0) {
        morton_code current_node = trial_nodes.back();
        trial_nodes.pop_back();

        if ((code_a < current_node) && (current_node < code_b) && morton_helper.is_ancestor(code_b, current_node)) {
          min_lin_tree.push_back(current_node);
        }
        else if (morton_helper.is_ancestor(nearest_common_ancestor, current_node)) {
          ippl::vector_t<morton_code> children = morton_helper.get_children(current_node); 
          for (morton_code& child : children) trial_nodes.push_back(child);
        }
      }

      std::sort(min_lin_tree.begin(), min_lin_tree.end());
      return min_lin_tree;
    }

    template<size_t Dim>
    ippl::vector_t<morton_code> OrthoTree<Dim>::linearise_octants(const ippl::vector_t<morton_code>& octants)
    {
        ippl::vector_t<morton_code> linearised;
        for(size_t i = 0; i < octants.size()-1; ++i)
        {
            if(morton_helper.is_ancestor(octants[i+1], octants[i]))
            {
                continue;
            }
            linearised.push_back(octants[i]);
        }
        linearised.push_back(octants.back());
        return linearised;
    }

    template<size_t Dim>
    void OrthoTree<Dim>::linearise_tree()
    {
        tree_m = linearise_octants(tree_m);
    }

} // namespace ippl