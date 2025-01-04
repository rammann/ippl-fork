#include "AidList.h"
#include <cstdint>
#include <mpi.h>
#include <random>
#include <chrono>

#include "Communicate/Window.h"


namespace ippl {
    template <size_t Dim>
    AidList<Dim>::AidList(size_t max_depth)
        : world_rank(Comm->rank())
        , world_size(Comm->size())
        , max_depth(max_depth)
        , morton_helper(Morton<Dim>(max_depth))
        , logger("AidList", std::cerr, INFORM_ALL_NODES) {
        logger.setOutputLevel(5);
        logger.setPrintNode(INFORM_ALL_NODES);
        bucket_borders = Kokkos::View<morton_code*>("bucket_borders", world_size - 1);

        // logger << "Initialized AidList" << endl;
    }

    template <size_t Dim>
    template <typename PLayout>
    void AidList<Dim>::initialize(const BoundingBox<Dim>& root_bounds, PLayout const& particles) {
        logger<<"Initializing AidList"<<endl;
        if (world_rank == 0) {
            initialize_from_rank(max_depth, root_bounds, particles);
            logger << "initial Aid list is initialized with size: " << octants.size() << endl;
            distribute_buckets();
        }
        else{
            mpi::Status stat;
            //receive the bucket borders

            Comm->broadcast(bucket_borders.data(), bucket_borders.size(), 0);

            //receive the bucket size
            size_t bucket_size;
            Comm->recv(&bucket_size, 1, 0, 1, stat);

            IpplTimings::TimerRef bucket_distribution = IpplTimings::getTimer("Bucket Distribution Timer");
            IpplTimings::startTimer(bucket_distribution);
            //allocate the space for the bucket in the aid list
            octants = Kokkos::View<morton_code*>("octants", bucket_size);
            particle_ids = Kokkos::View<size_t*>("particle_ids", bucket_size);
            
            //receive the octants and the particle ids
            logger << "Receiving octants and particle ids on rank " << world_rank << endl;
            if(bucket_size > 0){
                Comm->recv(octants.data(), bucket_size, 0, 0, stat);
                Comm->recv(particle_ids.data(), bucket_size, 0, 1, stat);
            }
            IpplTimings::stopTimer(bucket_distribution);

        }
        IpplTimings::TimerRef sort_aidlist = IpplTimings::getTimer("Sort AidList Timer");
        IpplTimings::startTimer(sort_aidlist);
        sort_local_aidlist();
        IpplTimings::stopTimer(sort_aidlist);
        logger << "AidList initialized with size: " << octants.size() << endl;
    }

    template <size_t Dim>
    void AidList<Dim>::distribute_buckets() {
        logger << "Distributing buckets" << endl;

        const size_t n_particles = octants.size();

        // vector storing the actual sizes of the buckets initially 0
        Kokkos::View<size_t*> bucket_sizes("bucket_sizes", world_size);
        Kokkos::deep_copy(bucket_sizes, 0);

        Kokkos::View<size_t*> buckets_particle_ids("buckets_particle_ids", n_particles);
        Kokkos::View<morton_code*> buckets_octants("buckets_octants", n_particles);
        Kokkos::View<size_t*> bucket_indices("bucket_indices", world_size);
        Kokkos::deep_copy(bucket_indices, 0);

        Kokkos::View<size_t*> sizes_prefix_sum("sizes_prefix_sum", world_size);

        // get the target rank for a given octant
        auto get_target_rank = [&](morton_code octant) {
            const size_t target_rank =
                std::upper_bound(bucket_borders.data(),
                                 bucket_borders.data() + bucket_borders.size(), octant)
                - bucket_borders.data();

            return target_rank;
        };

        /**
         * Calculate the bucket borders
         */
        {
            const morton_code max_octant = morton_helper.get_deepest_last_descendant(0);
            const size_t avg_bucket_size = max_octant / world_size;
            const size_t k               = max_octant % world_size;

            // distribute ranges of possible octants equally among the ranks
            // the first k ranks get a range one larger than the rest 
            // !!! These might not be valid morton_codes but it doesn't matter since
            // they are only used to distribute the octants
            for (size_t i = 1; i < world_size; ++i) {
                const size_t offset = i < k ? i : k;

                bucket_borders(i - 1) = i * avg_bucket_size + offset;
                logger << "actual Bucket border " << i << ": " << bucket_borders(i) << endl;
            }

            std::sort(bucket_borders.data(), bucket_borders.data() + bucket_borders.extent(0));
            Comm->broadcast(bucket_borders.data(), bucket_borders.size(), 0);
        }

        IpplTimings::TimerRef bucket_distribution = IpplTimings::getTimer("Bucket Distribution Timer");
        IpplTimings::startTimer(bucket_distribution);

        /**
         * Calculate bucket size for each rank and collect them as a prefix sum in sizes_prefix_sum
         */
        {
            for (unsigned i = 0; i < n_particles; i++) {
                const morton_code octant = octants(i);
                const size_t target_rank = get_target_rank(octant);
                bucket_sizes(target_rank)++;
            }

            size_t local_total = 0;
            Kokkos::parallel_scan(
                "prefix_sum", world_size,
                KOKKOS_LAMBDA(const size_t i, size_t& sum, const bool final) {
                    sum += bucket_sizes(i);
                    if (final) {
                        sizes_prefix_sum(i) = sum;
                    }
                },
                local_total);
        }

        /**
         * Populate the buckets for each rank
         */
        {
            for (unsigned i = 0; i < n_particles; i++) {
                const morton_code octant = octants(i);
                const size_t particle_id = particle_ids(i);

                const size_t target_rank  = get_target_rank(octant);
                const size_t target_index = sizes_prefix_sum(target_rank)
                                            - bucket_sizes(target_rank)
                                            + bucket_indices(target_rank);

                assert(bucket_indices(target_rank) < bucket_sizes(target_rank));

                buckets_octants(target_index)      = octant;
                buckets_particle_ids(target_index) = particle_id;

                bucket_indices(target_rank)++;
            }
        }

        /**
         * log the bucket sizes
         * TODO: remove this
         */
        {
            if (world_rank == 0) {
                for (size_t i = 0; i < world_size; ++i) {
                    logger << "Bucket " << i << " has size: " << bucket_sizes(i) << endl;
                }
            }
        }

        /**
         * Distribute the buckets to its corresponding rank
         */
        {
            size_t index = 0;
            for (size_t i = 1; i < world_size; ++i) {
                Comm->send(bucket_sizes(i), 1, i, 1);

                if (bucket_sizes(i) == 0) {
                    continue;
                }

                const size_t start_index = sizes_prefix_sum(i) - bucket_sizes(i);

                Comm->send(*(buckets_octants.data() + start_index), bucket_sizes(i), i, 0);
                Comm->send(*(buckets_particle_ids.data() + start_index), bucket_sizes(i), i, 1);
            }
        }

        /**
         * Set bucket on rank 0
         */
        {
            // resize aid_list storage on rank 0
            Kokkos::resize(octants, bucket_sizes(0));
            Kokkos::resize(particle_ids, bucket_sizes(0));

            IpplTimings::stopTimer(bucket_distribution);

            auto index_pair = std::make_pair((size_t)0, bucket_sizes(0));
            Kokkos::deep_copy(octants, Kokkos::subview(buckets_octants, index_pair));
            Kokkos::deep_copy(particle_ids, Kokkos::subview(buckets_particle_ids, index_pair));
        }
    }

    template <size_t Dim>
    void AidList<Dim>::sort_local_aidlist() {

        // sort the local aid list
        Kokkos::View<size_t*> indices("indices", octants.size());
        Kokkos::parallel_for("Sort indices", octants.size(), KOKKOS_LAMBDA(const size_t i) {
            indices(i) = i;
        });
        std::sort(indices.data(), indices.data() + indices.extent(0), [&](size_t a, size_t b) {
            return octants(a) < octants(b);
        });

        // allocate the space for the sorted aid list
        Kokkos::View<morton_code*> sorted_octants("sorted_octants", octants.size());
        Kokkos::View<size_t*> sorted_particle_ids("sorted_particle_ids", octants.size());

        // fill the sorted aid list
        Kokkos::parallel_for("Fill sorted aid list", octants.size(), KOKKOS_LAMBDA(const size_t i) {
            sorted_octants(i) = octants(indices(i));
            sorted_particle_ids(i) = particle_ids(indices(i));
        });

        // swap the sorted aid list with the original one
        octants = sorted_octants;
        particle_ids = sorted_particle_ids;
        return;
    }



    template <size_t Dim>
    template <typename PLayout>
    bool AidList<Dim>::is_gathered(ippl::ParticleBase<PLayout> const& particles) {
        return particles.getLocalNum() == particles.getTotalNum();
    }

    template <size_t Dim>
    void AidList<Dim>::initialize_from_rank(
        size_t max_depth, const BoundingBox<Dim>& root_bounds,
        OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
        if (world_rank != 0) {
            throw std::runtime_error("This function should only be called on rank 0!");
        }

        if (!this->is_gathered(particles)) {
            throw std::runtime_error(
                "can only initialize if all particles are gathered on one rank!");
        }

        const size_t n_particles = particles.getTotalNum();
        const size_t grid_size = (size_t(1) << max_depth);
        using real_coordinate = real_coordinate_template<Dim>;
        using grid_coordinate = grid_coordinate_template<Dim>;
        const real_coordinate root_bounds_size = root_bounds.get_max() - root_bounds.get_min();

        // allocate the space for the octants and the particle ids
        octants = Kokkos::View<morton_code *> ("octants", n_particles);
        particle_ids = Kokkos::View<size_t *> ("particle_ids", n_particles);

        Kokkos::parallel_for(
            "InitializeAidList", Kokkos::RangePolicy<>(0, n_particles),
            KOKKOS_LAMBDA(const size_t i) {
                // Calculate grid coordinate
                const grid_coordinate grid_coord = static_cast<grid_coordinate>(
                    (particles.R(i) - root_bounds.get_min()) * (grid_size - 1) / root_bounds_size);

                octants(i)      = morton_helper.encode(grid_coord, max_depth);
                particle_ids(i) = i;
            });

        //log size of aid list
        logger << "Size of aid list: " << octants.size() << endl;
    }

    template <size_t Dim>
    size_t AidList<Dim>::getLowerBoundIndex(morton_code target_octant) const {
        const auto lower_bound_it =
            std::lower_bound(octants.data(), octants.data() + octants.extent(0), target_octant,
                [](const morton_code& octants_entry, const morton_code& target) {
                    return octants_entry < target;
                });

        return static_cast<size_t>(lower_bound_it - this->octants.data());
    }

    template <size_t Dim>
    size_t AidList<Dim>::getUpperBoundIndexExclusive(morton_code target_octant) const {
        const auto upper_bound_it =
            std::upper_bound(octants.data(), octants.data() + octants.extent(0), target_octant,
                [](const morton_code& target, const morton_code& octants_entry) {
                    return target < octants_entry;
                });

        return static_cast<size_t>(upper_bound_it - this->octants.data());
    }

    template <size_t Dim>
    size_t AidList<Dim>::getUpperBoundIndexInclusive(morton_code target_octant) const {
        const size_t upper_bound_idx = getUpperBoundIndexExclusive(target_octant);
        return upper_bound_idx != 0 ? upper_bound_idx - 1 : upper_bound_idx;
    }

    template <size_t Dim>
    size_t AidList<Dim>::getNumParticlesInOctant(morton_code octant) const {
        const size_t lower_bound_idx = getLowerBoundIndex(octant);
        const size_t upper_bound_idx =
            getUpperBoundIndexExclusive(morton_helper.get_deepest_last_descendant(octant));

        if (lower_bound_idx > upper_bound_idx) {
            throw std::runtime_error("loweridx > upper_idx in getNumParticlesInOctant...");
        }

        return upper_bound_idx - lower_bound_idx;
    }

    template <size_t Dim>
    std::pair<morton_code, morton_code> AidList<Dim>::getMinReqOctants() {
        morton_code min_max_octants[2];

        if (octants.size() == 0) {
            min_max_octants[0] = 0;
            min_max_octants[1] = 0;
            return std::make_pair(min_max_octants[0], min_max_octants[1]);
        }

        min_max_octants[0] = octants(0);
        min_max_octants[1] = octants(octants.size() - 1);
        return std::make_pair(min_max_octants[0], min_max_octants[1]);
    }

    template <size_t Dim>
    void AidList<Dim>::innitFromOctants(morton_code min_octant, morton_code max_octant) {
        // holds min/max octant from each rank
        Kokkos::View<size_t*> ranges("ranges", 2 * world_size);
        // min/max indices of data we will send to other ranks
        Kokkos::View<size_t*> send_indices("send_indices", 2 * world_size);
        // min/max indices of octants we receive from each rank
        Kokkos::View<size_t*> recv_indices("recv_indices", 2 * world_size);

        // number of octants this rank will receive
        size_t new_size_after_exchange = 0;

        /**
         * Populate the ranges view with the min/max octant for each rank.
         */
        {
            auto ranges_begin = std::span(ranges.data(), ranges.size()).begin();
            Kokkos::deep_copy(ranges, 0);

            mpi::rma::Window<mpi::rma::Active> range_window;
            range_window.create(*Comm, ranges_begin, ranges_begin + ranges.size());
            range_window.fence(0);

            const morton_code dld_root     = morton_helper.get_deepest_last_descendant(0);
            morton_code lower_bound_octant = 0;
            morton_code upper_bound_octant = 0;
            for (size_t i = 0; i < world_size; ++i) {
                upper_bound_octant = dld_root;
                if (i < world_size - 1) {
                    upper_bound_octant = bucket_borders(i);
                }

                if (i > 0) {
                    lower_bound_octant = bucket_borders(i - 1);
                }

                // skip processor if no interesting octants are there
                if (upper_bound_octant < min_octant || lower_bound_octant >= max_octant) {
                    continue;
                }

                morton_code lower_range = std::max(min_octant, lower_bound_octant);
                morton_code upper_range = std::min(max_octant, upper_bound_octant);

                // no need to send to ourselves
                if (i == world_rank) {
                    ranges(2 * i)     = lower_range;
                    ranges(2 * i + 1) = upper_range;
                    continue;
                }

                // find the range of octants that are in the current bucket
                range_window.put(lower_range, i, 2 * world_rank);
                range_window.put(upper_range, i, 2 * world_rank + 1);
            }

            range_window.fence(0);
        }

        /**
         * Calculate the amount of octants we send and receive to/from each rank.
         * - populate: send_indices
         * - populate: recv_indices
         * - calcualte: new_size_after_exchange = total number of octants/P_ids this rank will
         * receive
         */
        {
            auto recv_indices_begin = std::span(recv_indices.data(), recv_indices.size()).begin();
            Kokkos::deep_copy(recv_indices, 0);

            mpi::rma::Window<mpi::rma::Active> idx_window;
            idx_window.create(*Comm, recv_indices_begin, recv_indices_begin + recv_indices.size());
            idx_window.fence(0);

            for (unsigned rank = 0; rank < world_size; ++rank) {
                /*
                 * Skip ranks where min_octant == max_octant
                 * this works since ranks whose octants are for example way 
                 * bigger than the ones here will have set min_octant = max_octant = 0
                 */
                if (ranges(2 * rank) == ranges(2 * rank + 1)) {
                    continue;
                }
                send_indices(2 * rank)     = getLowerBoundIndex(ranges(2 * rank));
                send_indices(2 * rank + 1) = getLowerBoundIndex(ranges(2 * rank + 1));

                size_t send_size = send_indices(2 * rank + 1) - send_indices(2 * rank);

                // no need to communicate with ourselves
                if (rank == world_rank) {
                    recv_indices(2 * rank)     = send_indices(2 * rank);
                    recv_indices(2 * rank + 1) = send_indices(2 * rank + 1);
                    continue;
                }

                auto indices_a = send_indices(2 * rank);
                auto indices_b = send_indices(2 * rank + 1);
                idx_window.put(indices_a, rank, 2 * world_rank);
                idx_window.put(indices_b, rank, 2 * world_rank + 1);
            }

            idx_window.fence(0);

            Kokkos::parallel_reduce(
                "compute new size", world_size,
                KOKKOS_LAMBDA(const size_t i, size_t& local_new_size) {
                    local_new_size += recv_indices(2 * i + 1) - recv_indices(2 * i);
                },
                new_size_after_exchange);
        }

        /**
         * Exchange the octants between ranks.
         */
        {
            Kokkos::View<morton_code*> new_octants("new_octants", new_size_after_exchange);
            Kokkos::View<size_t*> new_particle_ids("new_particle_ids", new_size_after_exchange);

            auto new_octants_start_it =
                std::span(new_octants.data(), new_size_after_exchange).begin();
            auto new_particles_start_it =
                std::span(new_particle_ids.data(), new_size_after_exchange).begin();
            auto octants_begin          = std::span(octants.data(), octants.size()).begin();
            auto particle_ids_begin = std::span(particle_ids.data(), particle_ids.size()).begin();

            mpi::rma::Window<mpi::rma::Active> octants_window;
            mpi::rma::Window<mpi::rma::Active> particle_ids_window;

            octants_window.create(*Comm, octants_begin, octants_begin + octants.size());
            particle_ids_window.create(*Comm, particle_ids_begin,
                                       particle_ids_begin + particle_ids.size());

            particle_ids_window.fence(0);
            octants_window.fence(0);

            size_t last_insert_idx = 0;
            for (unsigned rank = 0; rank < world_size; ++rank) {
                if (recv_indices(2 * rank) == recv_indices(2 * rank + 1)) {
                    continue;
                }

                size_t recv_size = recv_indices(2 * rank + 1) - recv_indices(2 * rank);
                assert(recv_size > 0);
                auto start_it_octants      = new_octants_start_it + last_insert_idx;
                auto end_it_octants        = start_it_octants + recv_size;
                auto start_it_particle_ids = new_particles_start_it + last_insert_idx;
                auto end_it_particle_ids   = start_it_particle_ids + recv_size;

                static_assert(std::contiguous_iterator<decltype(start_it_octants)>,
                              "Iterator does not satisfy contiguous_iterator");

                if (rank == world_rank) {
                    last_insert_idx += recv_size;
                    auto source_index_pair =
                        std::make_pair(recv_indices(2 * rank), recv_indices(2 * rank + 1));
                    auto source_subview = Kokkos::subview(octants, source_index_pair);

                    auto dest_index_pair =
                        std::make_pair(last_insert_idx - recv_size, last_insert_idx);
                    auto dest_subview = Kokkos::subview(new_octants, dest_index_pair);

                    Kokkos::deep_copy(dest_subview, source_subview);

                    auto source_index_pair_particles =
                        std::make_pair(recv_indices(2 * rank), recv_indices(2 * rank + 1));
                    auto source_subview_particles =
                        Kokkos::subview(particle_ids, source_index_pair_particles);

                    auto dest_index_pair_particles =
                        std::make_pair(last_insert_idx - recv_size, last_insert_idx);
                    auto dest_subview_particles =
                        Kokkos::subview(new_particle_ids, dest_index_pair_particles);

                    Kokkos::deep_copy(dest_subview_particles, source_subview_particles);
                    continue;
                }

                last_insert_idx += recv_size;
                octants_window.get(start_it_octants, end_it_octants, rank, recv_indices(2 * rank));
                particle_ids_window.get(start_it_particle_ids, end_it_particle_ids, rank,
                                        recv_indices(2 * rank));
            }

            octants_window.fence(0);
            particle_ids_window.fence(0);

            octants      = new_octants;
            particle_ids = new_particle_ids;
        }

        /**
         * Update the bucket borders on each rank.
         */
        {
            auto bucket_borders_begin =
                std::span(bucket_borders.data(), bucket_borders.size()).begin();

            mpi::rma::Window<mpi::rma::Active> bucket_window;
            bucket_window.create(*Comm, bucket_borders_begin,
                                 bucket_borders_begin + bucket_borders.size());
            bucket_window.fence(0);

            // update buckets
            if (world_rank != 0) {
                bucket_window.put(min_octant, 0, world_rank - 1);
            }
            bucket_window.fence(0);
            if (world_rank != 0) {
                bucket_window.get(bucket_borders_begin,
                                  bucket_borders_begin + bucket_borders.size(), 0, 0);
            }
            bucket_window.fence(0);
            if (world_rank == 0) {
                for (size_t i = 0; i < bucket_borders.size(); ++i) {
                    logger << "Bucket border " << i << ": " << bucket_borders(i) << endl;
                }
            }
        }
    }

    template <size_t Dim>
    template <typename Container>
    Kokkos::View<size_t*> AidList<Dim>::getNumParticlesInOctantsParallel(
        const Container& octant_container) {
        IpplTimings::TimerRef timer = IpplTimings::getTimer("getNumParticlesInOctantsParallel");
        IpplTimings::startTimer(timer);

        morton_code min_step = morton_helper.get_step_size(max_depth);
        morton_code min_octant = morton_helper.get_deepest_first_descendant(octant_container[0]);
        morton_code max_octant = morton_helper.get_deepest_last_descendant(octant_container[octant_container.size() - 1]) + min_step;
        innitFromOctants(min_octant, max_octant);

        Kokkos::View<size_t*> result("result", octant_container.size());
        size_t total_weight = 0;
        for (size_t i = 0; i < octant_container.size(); ++i) {
            result(i) = getNumParticlesInOctant(octant_container[i]);
            total_weight += result(i);
        }

        IpplTimings::stopTimer(timer);

        return result;
    }
}  // namespace ippl
