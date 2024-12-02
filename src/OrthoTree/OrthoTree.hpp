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

#define END_FUNC                   \
    LOG << "FINISHED\n\n" << endl; \
    --stack_depth

#define START_FUNC \
    ++stack_depth; \
    LOG << "STARTING\n\n" << endl

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
    Kokkos::View<morton_code*> OrthoTree<Dim>::build_tree(particle_t const& particles) {
        START_FUNC;

        world_size = Comm->size();
        world_rank = Comm->rank();  // TODO: move this to constructor, but then all tests need a
                                    // main to init ippl

        auto relevant_octants = get_relevant_aid_list(particles);
        /**
         * The relevant morton_codes to start the algorithm have been sent to the other ranks.
         * We can now start the actual algorithm.
         *
         * auto octants = block_partition();
         * -> those octants are basically line 7 in algo4
         */

        auto octants = block_partition(relevant_octants);
        init_aid_list_from_octants(octants);

        // Each proc has now as much of the aid_list as he needs and can start building the tree.
        Kokkos::View<morton_code*> tree_view = build_tree_from_octants(octants);

        Comm->barrier();
        logger.flush();

        {
            const size_t col_width = 15;

            auto printer = [col_width](const auto& rank, const auto& tree_size,
                                       const auto& num_particles) {
                std::cerr << std::left << std::setw(col_width) << rank << std::left
                          << std::setw(col_width) << tree_size << std::left << std::setw(col_width)
                          << num_particles << std::endl;
            };

            size_t total_particles = 0;
            for (size_t i = 0; i < tree_view.size(); ++i) {
                total_particles += get_num_particles_in_octant(tree_view[i]);
            }

            {
                int ring_buf;
                if (world_rank + 1 < world_size) {
                    mpi::Status status;
                    Comm->recv(&ring_buf, 1, world_rank + 1, 0, status);
                } else {
                    std::cerr << std::string(col_width * 3, '=') << std::endl;
                    printer("rank", "octants", "particles");
                    std::cerr << std::string(col_width * 3, '-') << std::endl;
                }

                printer(world_rank, tree_view.size(), total_particles);

                if (world_rank > 0) {
                    Comm->send(ring_buf, 1, world_rank - 1, 0);
                }
            }

            size_t global_total = 0;
            Comm->scan(&total_particles, &global_total, 1, std::plus<size_t>());

            if (world_rank == 0) {
                std::cerr << std::string(col_width * 3, '-') << std::endl
                          << "We now have " << global_total << " particles" << std::endl
                          << "(" << (100.0 * (double)global_total / (double)particles.getTotalNum())
                          << "% of starting value lol)" << std::endl
                          << std::string(col_width * 3, '-') << std::endl;
            }
        }

        return tree_view;
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::get_relevant_aid_list(particle_t const& particles) {
        START_FUNC;
        /**
         * This block initialised the aid list on rank 0 (assuming all particles are already
         * gathered on rank 0 beforehand).
         *
         * We then generate the AidList and split it into equally sized chuncks.
         * Each proc receivees the first and last octant in its chunk.
         */

        Kokkos::vector<morton_code> octant_buffer;
        if (world_rank == 0) {
            if (particles.getLocalNum() != particles.getTotalNum()) {
                throw std::runtime_error("particles must all be gathered on rank 0!");
            }

            this->aid_list = initialize_aid_list(particles);
            LOG << "aid list has size: " << aid_list.size() << endl;

            const size_t total_num_particles = particles.getTotalNum();
            const size_t batch_size          = total_num_particles / world_size;
            const size_t rank_0_size         = batch_size + (total_num_particles % batch_size);

            for (int iter_rank = 1; iter_rank < world_size; ++iter_rank) {
                const int start = ((iter_rank - 1) * batch_size) + rank_0_size;
                const int end   = start + batch_size;

                octant_buffer.clear();
                octant_buffer.push_back(aid_list[start].first);
                octant_buffer.push_back(aid_list[end - 1].first);

                LOG << "sending to rank " << iter_rank << ": " << octant_buffer[0] << ", "
                    << octant_buffer[1] << endl;

                try {
                    Comm->send(*octant_buffer.data(), 2, iter_rank, 0);
                } catch (const IpplException& e) {
                    LOG << "error during send in build_tree(): " << e.what() << endl;
                }
                LOG << "sent to rank " << iter_rank << endl;
            }

            octant_buffer.clear();
            octant_buffer.push_back(aid_list[0].first);
            octant_buffer.push_back(aid_list[rank_0_size - 1].first);
        } else {
            mpi::Status status;
            octant_buffer.clear();
            octant_buffer.resize(2);
            Comm->recv(octant_buffer.data(), 2, 0, 0, status);

            LOG << "received its octants: size=" << octant_buffer.size()
                << "with octants: " << octant_buffer[0] << ", " << octant_buffer[1] << endl;
        }

        END_FUNC;
        return octant_buffer;
    }

    template <size_t Dim>
    void OrthoTree<Dim>::init_aid_list_from_octants(const Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        /**
         * This if/else block corresponds to line 8 in algo4.
         * We send the relevant parts of the AidList to each processor so they can start
         * building the tree.
         */
        Kokkos::vector<morton_code> octant_buffer;
        Kokkos::vector<morton_code> id_buffer;
        size_t size_buff;
        if (world_rank == 0) {
            for (size_t rank = 1; rank < static_cast<size_t>(world_size); ++rank) {
                mpi::Status status;
                octant_buffer = Kokkos::vector<morton_code>(2);
                Comm->recv(octant_buffer.data(), 2, rank, 0, status);

                LOG << "received min/max octant from " << rank << endl;

                // get size of range in aid list
                auto lower_bound_idx = std::lower_bound(
                    this->aid_list.begin(), this->aid_list.end(), octant_buffer.front(),
                    [](const Kokkos::pair<unsigned long long, unsigned long>& pair,
                       const morton_code& val) {
                        return pair.first < val;
                    });

                auto upper_bound_idx = std::upper_bound(
                    this->aid_list.begin(), this->aid_list.end(), octant_buffer.back(),
                    [](const morton_code& val,
                       const Kokkos::pair<unsigned long long, unsigned long>& pair) {
                        return val < pair.first;
                    });

                // send size to rank
                size_t size_buff = static_cast<size_t>(upper_bound_idx - lower_bound_idx);
                Comm->send(size_buff, 1, rank, 0);
                LOG << "sent size to " << rank << endl;

                octant_buffer.clear();
                id_buffer.clear();
                for (auto it = lower_bound_idx; it != upper_bound_idx; ++it) {
                    octant_buffer.push_back(it->first);
                    id_buffer.push_back(it->second);
                }

                // send octantts
                // send particle ids
                Comm->send(*octant_buffer.data(), size_buff, rank, 0);
                Comm->send(*id_buffer.data(), size_buff, rank, 0);

                LOG << "sent buffers to " << rank << endl;
            }
        } else {
            // send own min/max octants
            octant_buffer.clear();
            octant_buffer.push_back(octants[0]);
            octant_buffer.push_back(morton_helper.get_deepest_last_descendant(octants.back()));
            Comm->send(*octant_buffer.data(), 2, 0, 0);

            LOG << "sent min/max octants" << endl;

            // receive aid list size
            mpi::Status status1;
            Comm->recv(&size_buff, 1, 0, 0, status1);
            LOG << "received size=" << size_buff << endl;
            // receive octants and receive ids
            octant_buffer.clear();
            octant_buffer.reserve(size_buff);
            id_buffer.clear();
            id_buffer.reserve(size_buff);

            mpi::Status status2, status3;
            Comm->recv(octant_buffer.data(), size_buff, 0, 0, status2);
            Comm->recv(id_buffer.data(), size_buff, 0, 0, status3);
            LOG << "received buffers" << endl;

            // copy octants and ids to own aid_list
            this->aid_list.clear();
            for (size_t i = 0; i < size_buff; ++i) {
                this->aid_list.push_back({octant_buffer[i], id_buffer[i]});
            }

            LOG << "DONE!" << endl;
        }

        END_FUNC;
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
            s.push({octants[i], get_num_particles_in_octant(octants[i])});
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
                const size_t count = get_num_particles_in_octant(child_octant);

                // no need to push in this case
                if (count > 0) {
                    s.push({child_octant, count});
                }
            }
        }

        // if we sort the tree after construction we can compare two trees
        std::sort(result_tree.begin(), result_tree.end());

        Kokkos::View<morton_code*> return_tree(result_tree.data());
        END_FUNC;
        return return_tree;
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::partition(Kokkos::vector<morton_code>& octants,
                                                          Kokkos::vector<size_t>& weights) {
        START_FUNC;
        world_rank = Comm->rank();
        world_size = Comm->size();

        // initialize the prefix_sum and the total weight
        Kokkos::vector<size_t> prefix_sum;
        prefix_sum.reserve(octants.size());
        size_t max = 0;

        // calculate the prefix_sum for the weight of each octant in octants
        //  get weight here later
        for (unsigned i = 0; i < octants.size(); ++i) {
            max += weights[i];
            prefix_sum.push_back(max);
        }

        // scan to get get the propper offsets for the prefix_sum
        size_t total;
        Comm->scan(&max, &total, 1, std::plus<morton_code>());

        // calculate the actual scan prefix_sum on each processor
        for (size_t i = 0; i < prefix_sum.size(); ++i) {
            prefix_sum[i] += total - max;
        }

        // broadcast the total weight to all processors
        Comm->broadcast<size_t>(&total, 1, world_size - 1);

        // initialize the average weight
        // might want to get a double? needs checking
        size_t avg_weight = total / world_size;
        size_t k          = total % world_size;

        Kokkos::vector<morton_code> total_octants;
        Kokkos::vector<mpi::Request> requests;
        Kokkos::vector<int> sizes(world_size);

        // BARRIER_LOG << "broadcasted total_weight =" << total << " with k=" << k << " and
        // avg_weight = " << avg_weight << std::endl;

        // initialize the start and end index for which processor receives which
        // local octants. Doing this here allows us to update these incrementally
        // in a two-pointer approach
        size_t start = 0, end = 0;
        // loop thorugh all processors

        for (size_t iter_rank = 1; iter_rank <= static_cast<size_t>(world_size); ++iter_rank) {
            if (iter_rank - 1
                == static_cast<size_t>(world_rank)) {  // no need to send data to myself
                continue;
            }

            // initialize the start and end index
            size_t startoffset = 0;
            size_t endoffset   = 0;

            // calculate the start and end offset for the processor in order
            // to distribute the remainder of total/world_size evenly
            if (iter_rank <= k) {
                startoffset = iter_rank - 1;
                endoffset   = iter_rank;
            } else {
                startoffset = k;
                endoffset   = k;
            }

            start = end;

            // calculate the start and end index for the processor
            for (size_t i = end; i < octants.size(); ++i) {
                if (prefix_sum[i] > avg_weight * (iter_rank - 1) + startoffset) {
                    start = i;
                    break;
                }
            }

            for (size_t i = start; i < octants.size(); ++i) {
                if (prefix_sum[i] > avg_weight * iter_rank + endoffset) {
                    end = i;
                    break;
                }
                if (i == octants.size() - 1) {
                    end = octants.size();
                }
            }

            // if the processor is the last one, add the remaining weight
            if (iter_rank == static_cast<size_t>(world_size) || end > octants.size()) {
                end = octants.size();
            }

            // initialize the new octants for the processor
            Kokkos::vector<morton_code> new_octants;

            // loop through the octants and add the octants to the new octants
            for (size_t i = start; i < end; ++i) {
                new_octants.push_back(octants[i]);
                total_octants.push_back(octants[i]);
            }

            LOG << "sending new octants size=" << new_octants.size() << " to rank=" << iter_rank - 1
                << endl;
            // send the number of new octants to the processor

            requests.push_back(mpi::Request());
            sizes[iter_rank - 1] = new_octants.size();
            Comm->isend((sizes[iter_rank - 1]), 1, iter_rank - 1, 1, requests[requests.size() - 1]);

            // send the new octants to the processor
            // Comm->isend(new_octants.size(), 1, p-1, 0, request1);

            requests.push_back(mpi::Request());
            if (sizes[iter_rank - 1] > 0) {
                Comm->isend<morton_code>(*new_octants.data(), new_octants.size(), iter_rank - 1, 0,
                                         requests[requests.size() - 1]);
            }
        }

        requests.clear();
        std::vector<mpi::Request> receives;
        // initialize the new octants for the current processor
        Kokkos::vector<morton_code> received_octants;
        for (size_t iter_rank = 0; iter_rank < static_cast<size_t>(world_size); ++iter_rank) {
            if (iter_rank == static_cast<size_t>(world_rank)) {
                continue;
            }

            int size;
            // receives.push_back(mpi::Request());
            //  receive the number of new octants
            mpi::Status stat;
            Comm->recv(&size, 1, iter_rank, 1, stat);
            // Comm->irecv(&size, 1, iter_rank, 0, receives);
            // initialize the new octants
            Kokkos::vector<morton_code> octants_buffer(size);
            // receive the new octants
            if (size > 0) {
                Comm->recv(octants_buffer.data(), size, iter_rank, 0, stat);
                // add the new octants to the received octants
                received_octants.insert(received_octants.end(), octants_buffer.begin(),
                                        octants_buffer.end());
            }
        }

        // BARRIER_LOG << " finished the second loop\n";

        // add the received octants to octants and remove total_octants
        Kokkos::vector<morton_code> partitioned_octants;
        unsigned int l1 = 0;
        unsigned int l2 = 0;
        unsigned int l3 = 0;

        while (l1 < octants.size()) {
            if (l2 == received_octants.size() || octants[l1] < received_octants[l2]) {
                if (l3 < total_octants.size() && octants[l1] == total_octants[l3]) {
                    l3++;
                } else {
                    partitioned_octants.push_back(octants[l1]);
                }
                l1++;
            } else {
                partitioned_octants.push_back(received_octants[l2]);
                l2++;
            }
        }

        while (l2 < received_octants.size()) {
            partitioned_octants.push_back(received_octants[l2]);
            l2++;
        }

        LOG << "num of octants on rank " << world_rank << " is " << partitioned_octants.size()
            << endl;

        END_FUNC;
        return partitioned_octants;
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
        START_FUNC;
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

        logger << "finished with num particles = "
               << static_cast<size_t>(upper_bound_idx - lower_bound_idx) << endl;
        END_FUNC;
        return static_cast<size_t>(upper_bound_idx - lower_bound_idx);
    }

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

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::linearise_octants(
        const Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        logger << "size: " << octants.size() << endl;
        Kokkos::vector<morton_code> linearised;

        for (size_t i = 0; i < octants.size() - 1; ++i) {
            if (morton_helper.is_ancestor(octants[i + 1], octants[i])) {
                continue;
            }

            linearised.push_back(octants[i]);
        }

        linearised.push_back(octants.back());

        logger << "finished, size is: " << linearised.size() << endl;
        END_FUNC;
        return linearised;
    }

    template<size_t Dim>
    void OrthoTree<Dim>::linearise_tree()
    {
        tree_m = linearise_octants(tree_m);
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::block_partition(
        Kokkos::vector<morton_code>& starting_octants) {
        START_FUNC;
        logger << "called with " << starting_octants.size() << " octants" << endl;

        Kokkos::vector<morton_code> T =
            complete_region(starting_octants.front(), starting_octants.back());

        logger << "T.size() = " << T.size() << endl;

        Kokkos::vector<morton_code> C;
        size_t lowest_level = std::numeric_limits<morton_code>::max();
        for (const morton_code& octant : T) {
            lowest_level = std::min(lowest_level, morton_helper.get_depth(octant));
        }

        for (morton_code octant : T) {
            if (morton_helper.get_depth(octant) == lowest_level) {
                C.push_back(octant);
            }
        }

        logger << "C.size()=" << C.size() << endl;
        Kokkos::vector<morton_code> G = complete_tree(C);
        logger << "we now have n_octants = " << G.size() << endl;

        Kokkos::vector<size_t> weights = get_num_particles_in_octants_parallel(G);
        logger << "weights have size: " << weights.size() << endl;
        /*
        for (size_t i = 0; i < G.size(); ++i) {
            morton_code base_tree_octant = G[i];
            weights[i]                   = std::count_if(
                starting_octants.begin(), starting_octants.end(),
                [&base_tree_octant, this](const morton_code& unpartitioned_tree_octant) {
                    return (unpartitioned_tree_octant == base_tree_octant)
                           || (morton_helper.is_descendant(unpartitioned_tree_octant,
                                                                             base_tree_octant));
                });
        }
        */

        Kokkos::vector<morton_code> partitioned_tree = partition(G, weights);

        // TODO: THIS MIGHT BE WRONG??
        Kokkos::vector<morton_code> global_unpartitioned_tree = starting_octants;
        starting_octants.clear();
        for (morton_code gup_octant : global_unpartitioned_tree) {
            for (const morton_code& p_octant : partitioned_tree) {
                if (gup_octant == p_octant || morton_helper.is_descendant(gup_octant, p_octant)) {
                    starting_octants.push_back(gup_octant);
                    break;
                }
            }
        }

        logger << "finished, partitioned_tree.size() = " << partitioned_tree.size() << endl;
        END_FUNC;
        return partitioned_tree;
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::complete_tree(
        Kokkos::vector<morton_code>& octants) {
        START_FUNC;
        world_rank = Comm->rank();
        world_size = Comm->size();

        logger << "called with n_octants=" << octants.size() << endl;
        // this removes duplicates, inefficient as of now
        std::map<morton_code, int> m;
        for ( auto octant : octants ) {
            ++m[octant];
        }

        logger << "map has " << m.size()
               << "octants in total, previously we had: " << octants.size() << endl;

        octants.clear();
        for ( const auto [octant, count] : m ) {
            octants.push_back(octant);
        }

        octants = linearise_octants(octants);

        Kokkos::vector<size_t> weights(octants.size(), 1);

        octants = partition(octants, weights);
        logger << "finished partition" << endl;

        if (octants.size() == 0) {
            throw std::runtime_error("SIZE IS ZERO HOW THE FUCK???");
        }

        morton_code first_rank0;
        if ( world_rank == 0 ) {
            const morton_code dfd_root = morton_helper.get_deepest_first_descendant(morton_code(0));
            const morton_code A_finest = morton_helper.get_nearest_common_ancestor(dfd_root, octants[0]);
            const morton_code first_child = morton_helper.get_first_child(A_finest);
            // this imitates push_front
            first_rank0 = first_child;
        }
        else if ( world_rank == world_size - 1 ) {
            const morton_code dld_root = morton_helper.get_deepest_last_descendant(morton_code(0));
            const morton_code A_finest = morton_helper.get_nearest_common_ancestor(dld_root, octants[0]);
            const morton_code last_child = morton_helper.get_last_child(A_finest);

            octants.push_back(last_child);
        }

        if ( world_rank > 0 ) {
            Comm->send(*octants.data(), 1, world_rank - 1, 0);
        }

        morton_code buff;
        if ( world_rank < world_size - 1 ) {
            mpi::Status status;
            Comm->recv(&buff, 1, world_rank + 1, 0, status);
            // do we need a status check here or not?
            octants.push_back(buff);
        }

        logger << "finished send/recv" << endl;

        Kokkos::vector<morton_code> R;
        // rank 0 works differently, as we need to 'simulate' push_front
        if ( world_rank == 0 ) {
            R.push_back(first_rank0);
            for (morton_code elem : complete_region(first_rank0, octants[0])) {
                R.push_back(elem);
            }
        }

        const size_t n = octants.size();
        for ( size_t i = 0; i < n - 1; ++i ) {
            for (morton_code elem : complete_region(octants[i], octants[i + 1])) {
                R.push_back(elem);
            }
            R.push_back(octants[i]);
        }

        if ( world_rank == world_size - 1 ) {
            R.push_back(octants[n - 1]);
        }

        logger << "finished" << endl;
        END_FUNC;
        return R;
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
        size_t num_octs = octants.size();
        Kokkos::vector<size_t> num_particles(num_octs);
        for (size_t i = 0; i < num_octs; ++i) {
            num_particles[i] = get_num_particles_in_octant(octants[i]);
        }
        return num_particles;
    }

}  // namespace ippl
