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

        auto [min_octant, max_octant] = get_relevant_aid_list(particles);
        /**
         * The relevant morton_codes to start the algorithm have been sent to the other ranks.
         * We can now start the actual algorithm.
         *
         * auto octants = block_partition();
         * -> those octants are basically line 7 in algo4
         */

        auto octants = block_partition(min_octant, max_octant);

        particles_to_file(particles);

        init_aid_list_from_octants(octants.front(), octants.back());

        // manually syncing full aid list
        /*
        size_t aid_list_size;
        if (world_rank == 0) {
            aid_list_size = this->aid_list.size();
        }

        Comm->broadcast(&aid_list_size, 1, 0);

        std::vector<morton_code> octant_buff, id_buff;
        if (world_rank == 0) {
            for (const auto& [octant, id] : this->aid_list) {
                octant_buff.push_back(octant);
                id_buff.push_back(id);
            }
            for (size_t rank = 1; rank < static_cast<size_t>(world_size); ++rank) {
                Comm->send(*octant_buff.data(), aid_list_size, rank, 0);
                Comm->send(*id_buff.data(), aid_list_size, rank, 0);
            }
        } else {
            octant_buff.resize(aid_list_size);
            id_buff.resize(aid_list_size);

            mpi::Status status1, status2;
            Comm->recv(octant_buff.data(), aid_list_size, 0, 0, status1);
            Comm->recv(id_buff.data(), aid_list_size, 0, 0, status1);

            this->aid_list.clear();
            for (size_t i = 0; i < aid_list_size; ++i) {
                aid_list.push_back({octant_buff[i], id_buff[i]});
            }
        }
        */

        // Each proc has now as much of the aid_list as he needs and can start building the
        // tree.
        Kokkos::View<morton_code*> tree_view = build_tree_from_octants(octants);

        Comm->barrier();
        logger.flush();

        size_t total_particles = 0;
        for (size_t i = 0; i < tree_view.size(); ++i) {
            total_particles += get_num_particles_in_octant(tree_view[i]);
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

        return tree_view;
    }

    template <size_t Dim>
    std::pair<morton_code, morton_code> OrthoTree<Dim>::get_relevant_aid_list(
        particle_t const& particles) {
        START_FUNC;
        /**
         * This block initialised the aid list on rank 0 (assuming all particles are already
         * gathered on rank 0 beforehand).
         *
         * We then generate the AidList and split it into equally sized chuncks.
         * Each proc receivees the first and last octant in its chunk.
         */

        Kokkos::vector<morton_code> octant_buffer;
        morton_code min_octant, max_octant;

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
            min_octant = aid_list[0].first;
            max_octant = aid_list[rank_0_size - 1].first;
        } else {
            mpi::Status status;
            octant_buffer.clear();
            octant_buffer.resize(2);
            Comm->recv(octant_buffer.data(), 2, 0, 0, status);

            min_octant = octant_buffer[0];
            max_octant = octant_buffer[1];

            LOG << "received its octants: size=" << octant_buffer.size()
                << "with octants: " << octant_buffer[0] << ", " << octant_buffer[1] << endl;
        }

        if (min_octant >= max_octant) {
            throw std::runtime_error("this shouldnt happen (get_relevant_aid_list)");
        }

        END_FUNC;
        return {min_octant, max_octant};
    }

    template <size_t Dim>
    void OrthoTree<Dim>::init_aid_list_from_octants(morton_code min_octant,
                                                    morton_code max_octant) {
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

                // const size_t lower_bound_idx = getAidList_lowerBound(
                // morton_helper.get_deepest_first_descendant(octant_buffer.front()));

                // const size_t upper_bound_idx =
                // getAidList_lowerBound(morton_helper.get_deepest_last_descendant(octant_buffer.back()));

                auto lower_bound_it = std::lower_bound(
                    this->aid_list.begin(), this->aid_list.end(),
                    morton_helper.get_deepest_first_descendant(octant_buffer.front()),
                    [](const auto& pair, const morton_code& val) {
                        return pair.first < val;
                    });

                auto upper_bound_it = std::upper_bound(
                    this->aid_list.begin(), this->aid_list.end(),
                    morton_helper.get_deepest_last_descendant(octant_buffer.back()),
                    [](const morton_code& val, const auto& pair) {
                        return val < pair.first;
                    });

                // send size to rank
                size_t size_buff = static_cast<size_t>(upper_bound_it - lower_bound_it);
                Comm->send(size_buff, 1, rank, 0);
                LOG << "sent size to " << rank << endl;

                octant_buffer.clear();
                id_buffer.clear();
                for (auto it = lower_bound_it; it != upper_bound_it; ++it) {
                    octant_buffer.push_back(it->first);
                    id_buffer.push_back(it->second);
                }

                // send octantts
                // send particle ids
                Comm->send(*octant_buffer.data(), size_buff, rank, 0);
                Comm->send(*id_buffer.data(), size_buff, rank, 0);

                LOG << "sent buffers to " << rank << endl;
            }

            // REDUCING OWN AID LIST (RANK 0) TODO: MAYBE REMOVE THIS?
            // const size_t lower_bound_idx =
            // getAidList_lowerBound(morton_helper.get_deepest_first_descendant(min_octant));

            // const size_t upper_bound_idx =
            // getAidList_lowerBound(morton_helper.get_deepest_last_descendant(max_octant));

            auto lower_bound_it =
                std::lower_bound(this->aid_list.begin(), this->aid_list.end(),
                                 morton_helper.get_deepest_first_descendant(min_octant),
                                 [](const auto& pair, const morton_code& val) {
                                     return pair.first < val;
                                 });

            auto upper_bound_it =
                std::upper_bound(this->aid_list.begin(), this->aid_list.end(),
                                 morton_helper.get_deepest_last_descendant(max_octant),
                                 [](const morton_code& val, const auto& pair) {
                                     return val < pair.first;
                                 });

            aid_list_t own_aid_list;
            for (auto it = lower_bound_it; it != upper_bound_it; ++it) {
                own_aid_list.push_back(*it);
            }

            this->aid_list = own_aid_list;

        } else {
            // send own min/max octants
            octant_buffer.clear();
            octant_buffer.push_back(min_octant);
            octant_buffer.push_back(max_octant);
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

            std::sort(this->aid_list.begin(), this->aid_list.end(),
                      [](const auto& a, const auto& b) {
                          return a.first < b.first;
                      });

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
}  // namespace ippl