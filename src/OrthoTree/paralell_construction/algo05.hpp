#include "../OrthoTree.h"

/*
TODO:
- IMPLEMENT THIS NEW FUNCTION SIGNATURE
- WRITE TESTS FOR THE FUNCTION
- ADJUST THE SIGNATURE IN ORTHOTREE.H

namespace ippl {

    // entry with weights
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::partition(Kokkos::View<morton_code*> octants,
                                                          Kokkos::View<morton_code*> weights);

    // entry without weights
    template <size_t Dim>
    Kokkos::View<morton_code*> OrthoTree<Dim>::partition(Kokkos::View<morton_code*> octants);
}  // namespace ippl
*/

namespace ippl {
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
}  // namespace ippl