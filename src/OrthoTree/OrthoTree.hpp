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
        initialize_aid_list(particles);

        std::stack<std::pair<morton_code, size_t>> s;
        s.push({ morton_code(0), particles.getLocalNum() });

        while ( !s.empty() ) {
            const auto& [octant, count] = s.top(); s.pop();

            if ( count <= max_particles_per_node_m || morton_helper.get_depth(octant) >= max_depth_m ) {
                tree_m.push_back(octant);
                continue;
            }

            for ( const auto& child_code : morton_helper.get_children(octant) ) {
                size_t count = 0;
                for ( const auto& [particle_code, id] : aid_list ) {
                    if ( morton_helper.is_descendant(particle_code, child_code) ) {
                        ++count;
                    }
                }

                if ( count > 0 ) {
                    s.push({ child_code, count });
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

} // namespace ippl