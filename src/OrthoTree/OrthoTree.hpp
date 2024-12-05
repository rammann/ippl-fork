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
        this->aid_list = initialize_aid_list(particles);

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
                const size_t count = get_num_particles_in_octant(child_octant);

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

    template <size_t Dim>
    Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>> OrthoTree<Dim>::get_tree()
        const {
        START_FUNC;
        Kokkos::vector<Kokkos::pair<morton_code, Kokkos::vector<size_t>>> result;
        result.reserve(tree_m.size());
        for (auto octant : tree_m) {
            Kokkos::vector<size_t> particle_ids;
            for (const auto& [particle_code, id] : aid_list) {
                if (morton_helper.is_descendant(particle_code, octant)) {
                    particle_ids.push_back(id);
                }
            }

            result.push_back(Kokkos::make_pair(octant, particle_ids));
        }

        END_FUNC;
        return result;
    }

    template <size_t Dim>
    OrthoTree<Dim>::aid_list_t OrthoTree<Dim>::initialize_aid_list(particle_t const& particles) {
        START_FUNC;
        logger << "called with " << particles.getLocalNum() << " particles" << endl;
        // maybe get getGlobalNum() in the future?
        n_particles = particles.getLocalNum();
        const size_t grid_size = (size_t(1) << max_depth_m);

        // store dimensions of root bounding box
        const real_coordinate root_bounds_size = root_bounds_m.get_max() - root_bounds_m.get_min();

        aid_list_t ret_aid_list;
        ret_aid_list.resize(n_particles);

        for ( size_t i = 0; i < n_particles; ++i ) {
            // normalize particle coordinate inside the grid
            // particle locations are accessed with .R(index)
            const real_coordinate normalized = (particles.R(i) - root_bounds_m.get_min()) / root_bounds_size;

            // calculate the grid coordinate relative to the bounding box and grid size
            const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * grid_size);
            ret_aid_list[i]                  = {morton_helper.encode(grid_coord, max_depth_m), i};
        }

        // list is sorted by asccending morton codes
        std::sort(ret_aid_list.begin(), ret_aid_list.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        logger << "finished initialising the aid_list" << endl;
        END_FUNC;
        return ret_aid_list;
    }

    template <size_t Dim>
    size_t OrthoTree<Dim>::get_num_particles_in_octant(morton_code octant) {
        // No need to START_FUNC and END_FUNC here unless you want detailed logs

        const morton_code lower_bound_target = octant;
        const morton_code upper_bound_target = octant + morton_helper.get_step_size(octant);

        auto lower_bound_it =
            std::lower_bound(this->aid_list.begin(), this->aid_list.end(), lower_bound_target,
                             [](const auto& pair, const morton_code& val) {
                                 return pair.first < val;
                             });

        auto upper_bound_it =
            std::upper_bound(this->aid_list.begin(), this->aid_list.end(), upper_bound_target,
                             [](const morton_code& val, const auto& pair) {
                                 return val < pair.first;
                             });

        return static_cast<size_t>(upper_bound_it - lower_bound_it);
    }

    template <size_t Dim>
    Kokkos::vector<size_t> OrthoTree<Dim>::get_num_particles_in_octants_parallel(
        const Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        world_rank = Comm->rank();
        world_size = Comm->size();

        mpi::Status stat;
        Kokkos::vector<size_t> num_particles;

        if (world_rank == 0) {
            for (size_t rank = 1; rank < static_cast<size_t>(world_size); ++rank) {
                int req_size;
                Comm->recv(req_size, 1, rank, 1, stat);

                Kokkos::vector<morton_code> octants_buffer(req_size);
                Comm->recv(octants_buffer.data(), req_size, rank, 0, stat);

                Kokkos::vector<size_t> count_num_particles =
                    get_num_particles_in_octants_seqential(octants_buffer);

                Comm->send(*count_num_particles.data(), req_size, rank, 0);
            }

            // get own num_particles
            num_particles = get_num_particles_in_octants_seqential(octants);
        } else {
            // send own octants to rank 0
            int req_size = octants.size();
            Comm->send(req_size, 1, 0, 1);
            Comm->send(*octants.data(), octants.size(), 0, 0);

            // receive weight of each octant
            num_particles.clear();
            num_particles.resize(req_size);
            Comm->recv(num_particles.data(), req_size, 0, 0, stat);
        }

        logger << "finished, num_particles.size() = " << num_particles.size() << endl;
        END_FUNC;
        return num_particles;
    }

    template <size_t Dim>
    Kokkos::vector<size_t> OrthoTree<Dim>::get_num_particles_in_octants_seqential(
        const Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        size_t num_octs = octants.size();
        Kokkos::vector<size_t> num_particles(num_octs);
        for (size_t i = 0; i < num_octs; ++i) {
            num_particles[i] = get_num_particles_in_octant(octants[i]);
        }
        END_FUNC;
        return num_particles;
    }

#pragma region helpers
    template <size_t Dim>
    size_t OrthoTree<Dim>::getAidList_lowerBound(morton_code octant) {
        if (this->aid_list.size() == 0) {
            std::cerr << "AID LIST ON RANK " << Comm->rank() << " HAS NOT BEEN INITIALISED!";
            throw std::runtime_error("AID LIST ON RANK " + std::to_string(Comm->rank())
                                     + " HAS NOT BEEN INITIALISED!");
        }

        auto lower_bound_it = std::lower_bound(this->aid_list.begin(), this->aid_list.end(), octant,
                                               [](const auto& pair, const morton_code& val) {
                                                   return pair.first < val;
                                               });

        return static_cast<size_t>(lower_bound_it - this->aid_list.begin());
    }

    template <size_t Dim>
    size_t OrthoTree<Dim>::getAidList_upperBound(morton_code octant) {
        if (this->aid_list.size() == 0) {
            std::cerr << "AID LIST ON RANK " << Comm->rank() << " HAS NOT BEEN INITIALISED!";
            throw std::runtime_error("AID LIST ON RANK " + std::to_string(Comm->rank())
                                     + " HAS NOT BEEN INITIALISED!");
        }

        auto upper_bound_it = std::upper_bound(this->aid_list.begin(), this->aid_list.end(), octant,
                                               [](const morton_code& val, const auto& pair) {
                                                   return val < pair.first;
                                               });

        return static_cast<size_t>(upper_bound_it - this->aid_list.begin());
    }
#pragma endregion  // helpers
}  // namespace ippl
