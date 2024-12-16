#include "AidList.h"
#include "random"

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
        if (world_rank == 0) {
            initialize_from_rank(max_depth, root_bounds, particles);
            logger << "Aid list is initialized with size: " << size() << endl;
            distribute_buckets();
        }
        else{
            //receive the bucket borders
            Comm->broadcast(bucket_borders.data(), bucket_borders.extent(0), 0);

            //receive the bucket size
            size_t bucket_size;
            Comm->recv(&bucket_size, 0, 1);

            //allocate the space for the bucket in the aid list
            octants = Kokkos::View<morton_code*>("octants", bucket_size);
            particle_ids = Kokkos::View<size_t*>("particle_ids", bucket_size);

            //receive the octants and the particle ids
            Comm->recv(octants.data(), bucket_size, 0, 0);
            Comm->recv(particle_ids.data(), bucket_size, 0, 0);

            logger << "Received " << bucket_size << " octants on rank " << world_rank << endl;
        }
        sort_local_aidlist();
    }

    template <size_t Dim>
    void AidList<Dim>::distribute_buckets() {

        size_t n_particles = octants.size();

        for(size_t i = 0; i < world_size - 1; ++i) {

            bool is_unique = true;
            size_t index = 0;
            do{
                index = std::rand() % n_particles;
                for(size_t j = 0; j < i; ++j) {
                    if(bucket_borders(j) == octants(index)) {
                        is_unique = false;
                        break;
                    }
                }
            }while(!is_unique);
            bucket_borders(i) = getOctant((i + 1) * size() / world_size);
        }

        // sort the bucket borders
        std::sort(bucket_borders.data(), bucket_borders.data() + bucket_borders.extent(0));

        // broadcast the bucket borders
        Comm->broadcast(bucket_borders.data(), bucket_borders.extent(0), 0);

        // view of views for the buckets
        Kokkos::View<Kokkos::View<morton_code*>*> buckets_octants("buckets_octants", world_size);
        Kokkos::View<Kokkos::View<size_t*>*> buckets_particle_ids("buckets_particle_ids", world_size);

        // allocate the guesstimated space for the buckets
        for(size_t i = 0; i < world_size; ++i) {
            buckets(i) = Kokkos::View<morton_code*>("bucket_octants_" + std::to_string(i), n_particles/world_size);
            buckets_particle_ids(i) = Kokkos::View<size_t*>("bucket_particle_ids_" + std::to_string(i), n_particles/world_size);
        }

        //vector storing the actual sizes of the buckets initially 0
        Kokkos::View<size_t*> bucket_sizes("bucket_sizes", world_size);
        Kokkos::deep_copy(bucket_sizes, 0);


        //get the target rank for a given octant
        auto get_target_rank = [&](morton_code octant) {
            size_t target_rank = 0;
            for(size_t i = 0; i < world_size - 1; ++i) {
                if(octant < bucket_borders(i)) {
                    target_rank = i;
                    break;
                }
            }
            return target_rank;
        };

        // fill the buckets
        Kokkos::parallel_for("Fill buckets", n_particles, KOKKOS_LAMBDA(const size_t i) {
            const morton_code octant = octants(i);
            const size_t target_rank = get_target_rank(octant);
            const size_t idx = Kokkos::atomic_fetch_add(&bucket_sizes(target_rank), 1);
            //if the bucket is full, we need to resize it
            if(idx >= buckets(target_rank).extent(0)) {
                Kokkos::resize(buckets(target_rank), 2 * idx);
            }
            buckets_octants(target_rank)(idx) = octant;
            buckets_particle_ids(target_rank)(idx) = particle_ids(i);
        });

        // send the buckets

        for(size_t i = 1; i < world_size; ++i) {
            if(i == world_rank) {
                continue;
            }
            Comm->send(bucket_sizes(i), i, 1);
            Comm->send(buckets_octants(i).data(), bucket_sizes(i), i, 0);
            Comm->send(buckets_particle_ids(i).data(), bucket_sizes(i), i, 1);
        }

        // set the local bucket
        octants = buckets_octants(0);
        particle_ids = buckets_particle_ids(0);
        return;
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

        const size_t n_particles               = particles.getTotalNum();
        const size_t grid_size                 = (size_t(1) << max_depth);
        using real_coordinate                  = real_coordinate_template<Dim>;
        using grid_coordinate                  = grid_coordinate_template<Dim>;
        const real_coordinate root_bounds_size = root_bounds.get_max() - root_bounds.get_min();

        // allocate the space for the octants and the particle ids
        octants = Kokkos::View<morton_code *> ("octants", n_particles);
        particle_ids = Kokkos::View<size_t *> ("particle_ids", n_particles);

        for (size_t i = 0; i < n_particles; ++i) {
            // this gets rid of cancellation, thank you @NumCSE script
            const grid_coordinate grid_coord = static_cast<grid_coordinate>(
                (particles.R(i) - root_bounds.get_min()) * (grid_size - 1) / root_bounds_size);

            // encode the grid coordinate and store it
            octants(i) = morton_helper.encode(grid_coord,max_depth);
            // store the particle id
            particle_ids(i) = i;
        }
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
        if (world_rank == 0) {
            const size_t size       = this->size();
            const size_t batch_size = size / world_size;
            const size_t remainder  = size % world_size;

            // rank_{i+1} = [ (i * batch_size) + remainder, (i+1) * batch_size ]
            Kokkos::parallel_for("Send min/max octants", world_size - 1, [=, this](const size_t i) {
                morton_code local_min_max_octants[2] = {0};  // inside the parallel for because of
                                                             // read-only
                const size_t target_rank = i + 1;

                // distribute the remainders for a more balanced load
                const size_t start = (target_rank * batch_size) + (target_rank < remainder ? 1 : 0);
                const size_t end   = start + batch_size;

                local_min_max_octants[0] = getOctant(start);
                local_min_max_octants[1] = getOctant(end - 1);

                Comm->send(*local_min_max_octants, 2, target_rank, 0);
            });

            const size_t start = 0;
            const size_t end   = batch_size + (0 < remainder ? 1 : 0);  // one extra octant

            min_max_octants[0] = getOctant(start);
            min_max_octants[1] = getOctant(end - 1);
        } else {
            mpi::Status status;
            Comm->recv(min_max_octants, 2, 0, 0, status);
        }

        return std::make_pair(min_max_octants[0], min_max_octants[1]);
    }

    template <size_t Dim>
    void AidList<Dim>::innitFromOctants(morton_code min_octant, morton_code max_octant) {
        size_t size_buff;
        morton_code min_max_buff[2];
        if (world_rank == 0) {
            for (size_t rank = 1; rank < world_size; ++rank) {
                mpi::Status status;
                Comm->recv(*min_max_buff, 2, rank, 0, status);

                const size_t start = getLowerBoundIndex(min_max_buff[0]);
                const size_t end   = getUpperBoundIndexExclusive(min_max_buff[1]);
                size_buff          = end - start;

                Comm->send(size_buff, 1, rank, 0);

                Comm->send(*(octants.data() + start), size_buff, rank, 0);
                Comm->send(*(particle_ids.data() + start), size_buff, rank, 0);

                logger << "sent " << size_buff << " octants to rank " << rank << " range: ("
                       << getOctant(start) << ", " << getOctant(end) << ")" << endl;
            }
        } else {
            min_max_buff[0] = min_octant;
            min_max_buff[1] = morton_helper.get_deepest_last_descendant(max_octant);

            Comm->send(*min_max_buff, 2, 0, 0);

            mpi::Status size_status;
            Comm->recv(&size_buff, 1, 0, 0, size_status);

            this->octants      = Kokkos::View<morton_code*>("aid_list_octants", size_buff);
            this->particle_ids = Kokkos::View<size_t*>("aid_list_particle_ids", size_buff);

            mpi::Status octants_status, particle_id_status;
            Comm->recv(octants.data(), size_buff, 0, 0, octants_status);
            Comm->recv(particle_ids.data(), size_buff, 0, 0, particle_id_status);
        }
    }

    template <size_t Dim>
    template <typename Container>
    Kokkos::View<size_t*> AidList<Dim>::getNumParticlesInOctantsParallel(
        const Container& octant_container) {
        size_t size_buff;
        Kokkos::View<size_t*> weights("weights_view", octant_container.size());
        if (world_rank == 0) {
            for (size_t rank = 1; rank < world_size; ++rank) {
                // receive the size of the octants from rank
                mpi::Status size_status;
                Comm->recv(&size_buff, 1, rank, 0, size_status);

                // allocate the required space and receive the cotants
                std::vector<morton_code> octants_buff(size_buff);
                mpi::Status octants_status;
                Comm->recv(octants_buff.data(), size_buff, rank, 0, octants_status);

                // no need to shrink, we only send what we need
                if (weights.size() < size_buff) {
                    Kokkos::resize(weights, size_buff);
                }

                for (size_t i = 0; i < size_buff; ++i) {
                    weights[i] = getNumParticlesInOctant(octants_buff[i]);
                }

                // send back the weights
                Comm->send(*weights.data(), size_buff, rank, 0);
            }

            // calculate own weights
            size_buff = octant_container.size();
            Kokkos::resize(weights,
                           size_buff);  // we must shrink here, as we return wrong values if we dont
            for (size_t i = 0; i < size_buff; ++i) {
                weights[i] = getNumParticlesInOctant(octant_container[i]);
            }

        } else {
            // send size of octants we are requesting to rank 0
            size_buff = octant_container.size();
            Comm->send(size_buff, 1, 0, 0);
            // send the actual octants to rank 0
            Comm->send(*octant_container.data(), size_buff, 0, 0);
            // reserve space for the weights we will receive
            mpi::Status weights_status;
            Comm->recv(weights.data(), size_buff, 0, 0, weights_status);
        }

        return weights;
    }
}  // namespace ippl