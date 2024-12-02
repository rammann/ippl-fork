#include <ostream>

#include "OrthoTree/OrthoTreeTypes.h"

#include "Communicate/Communicator.h"
#include "Communicate/Operations.h"
#include "Communicate/Request.h"
#include "Kokkos_Vector.hpp"
#include "OrthoTree.h"

namespace ippl {

#define ENABLE_LOGGING_STUFF

#ifdef ENABLE_LOGGING_STUFF
    int call_depth = 0;
#define DEPTH_THINGY call_depth * 2
#define LOG                                                                                    \
    std::cerr << std::string(DEPTH_THINGY, ' ') << "RANK " << world_rank << " in " << __func__ \
              << ": "

#define BARRIER_LOG                                                                      \
    Comm->barrier();                                                                     \
    if (world_rank == 0)                                                                 \
    std::cerr << "\u001b[34m" << std::string(DEPTH_THINGY, ' ') << "reached barrier in " \
              << __func__ << ":\u001b[0m "

#define START_BARRIER                                                                              \
    do {                                                                                           \
        Comm->barrier();                                                                           \
        ++call_depth;                                                                              \
        if (world_rank == 0)                                                                       \
            std::cerr << std::endl                                                                 \
                      << std::string(DEPTH_THINGY, ' ') << "T" << std::string(25, '=')             \
                      << std::endl                                                                 \
                      << "\u001b[34m" << std::string(DEPTH_THINGY, ' ') << "starting " << __func__ \
                      << ":\u001b[0m " << std::endl;                                               \
    } while (0)

#define END_BARRIER                                                                                \
    do {                                                                                           \
        Comm->barrier();                                                                           \
        if (world_rank == 0)                                                                       \
            std::cerr << "\u001b[34m" << std::string(DEPTH_THINGY, ' ') << "finished " << __func__ \
                      << ":\u001b[0m" << std::endl                                                 \
                      << std::string(DEPTH_THINGY, ' ') << "L" << std::string(25, '=')             \
                      << std::endl                                                                 \
                      << std::endl;                                                                \
        --call_depth;                                                                              \
    } while (0)

#else
    class NullStream : public std::ostream {
    public:
        NullStream()
            : std::ostream(nullptr) {}

        template <typename T>
        NullStream& operator<<(const T&) {
            return *this;
        }

        // Overload for manipulators like std::endl
        NullStream& operator<<(std::ostream& (*)(std::ostream&)) { return *this; }
    };

    inline NullStream null_stream;

#define DEPTH_THINGY  0
#define LOG           null_stream
#define BARRIER_LOG   null_stream
#define START_BARRIER (void)0
#define END_BARRIER   (void)0
#endif

    template <size_t Dim>
    OrthoTree<Dim>::OrthoTree(size_t max_depth, size_t max_particles_per_node,
                              const bounds_t& root_bounds)
        : max_depth_m(max_depth)
        , max_particles_per_node_m(max_particles_per_node)
        , root_bounds_m(root_bounds)
        , morton_helper(max_depth) {}

    template <size_t Dim>
    void OrthoTree<Dim>::build_tree_naive(particle_t const& particles)
    {
        // this needs to be initialized before constructing the tree
        this->aid_list = initialize_aid_list(particles);

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
    Kokkos::vector<morton_code> OrthoTree<Dim>::build_tree(particle_t const& particles) {
        Kokkos::vector<morton_code> octant_buffer;

        world_rank = Comm->rank();
        world_size = Comm->size();

        if (world_rank == 0) {
            aid_list = initialize_aid_list(particles);
            LOG << "aid list has size: " << aid_list.size() << std::endl;
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
                    << octant_buffer[1] << std::endl;

                try {
                    Comm->send(*octant_buffer.data(), 2, iter_rank, 0);
                } catch (const IpplException& e) {
                    std::cerr << "error during send in build_tree(): " << e.what() << std::endl;
                }
                LOG << "sent to rank " << iter_rank << std::endl;
            }

            octant_buffer.clear();
            octant_buffer.push_back(aid_list[0].first);
            octant_buffer.push_back(aid_list[rank_0_size - 1].first);
        } else {
            mpi::Status status;
            octant_buffer = Kokkos::vector<morton_code>(2);  // TODO: how can this be done nicely
            try {
                Comm->recv(octant_buffer.data(), 2, 0, 0, status);
            } catch (const IpplException& e) {
                std::cerr << "error during receive in build_tree(): " << e.what() << std::endl;
            }

            LOG << "received its octants: size=" << octant_buffer.size()
                << "with octants: " << octant_buffer[0] << ", " << octant_buffer[1] << std::endl;
            assert(octant_buffer.size() == 2 && "we should have exactly two octants");
        }

        BARRIER_LOG << "done spreading aid_list, entering block_partition\n";
        LOG << "calling block_partition with: " << octant_buffer.size() << " octants\n";
        auto octants = block_partition(octant_buffer);
        //  todo build tree here

        std::cerr << "octants.size() = " << octants.size() << std::endl;

        return {};
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::partition(Kokkos::vector<morton_code>& octants, Kokkos::vector<size_t>& weights) {
        // START_BARRIER;
        LOG << "called with n_octants=" << octants.size() << std::endl;

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

        // BARRIER_LOG << "max=" << max << std::endl;

        // calculate the actual scan prefix_sum on each processor
        for (size_t i = 0; i < prefix_sum.size(); ++i) {
            prefix_sum[i] += total - max;
        }

      //broadcast the total weight to all processors
      Comm->broadcast<size_t>(&total, 1, world_size - 1);

      //initialize the average weight
      //might want to get a double? needs checking
      size_t avg_weight = total / world_size;
      size_t k          = total % world_size;

      Kokkos::vector<morton_code> total_octants;
      Kokkos::vector<mpi::Request> requests;
      Kokkos::vector<int> sizes(world_size);

      // BARRIER_LOG << "broadcasted total_weight =" << total << " with k=" << k << " and avg_weight
      // = " << avg_weight << std::endl;

      // initialize the start and end index for which processor receives which
      // local octants. Doing this here allows us to update these incrementally
      // in a two-pointer approach
      size_t start = 0, end = 0;
      //loop thorugh all processors

      // BARRIER_LOG << "starting first loop\n";
      for (size_t iter_rank = 1; iter_rank <= static_cast<size_t>(world_size); ++iter_rank) {
          if (iter_rank - 1 == static_cast<size_t>(world_rank)) {  // no need to send data to myself
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
              << std::endl;
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

      LOG << "exited the first loop" << std::endl;
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

      //add the received octants to octants and remove total_octants
      Kokkos::vector<morton_code> partitioned_octants;
      unsigned int l1 = 0;
      unsigned int l2 = 0;
      unsigned int l3 = 0;

      while(l1 < octants.size()){
        if(l2 == received_octants.size() || octants[l1] < received_octants[l2]){
          if (l3 < total_octants.size() && octants[l1] == total_octants[l3]){
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

      while (l2 < received_octants.size()){
        partitioned_octants.push_back(received_octants[l2]);
        l2++;
      }

      //std::cout << "num of received octants on rank " << rank << " is " << received_octants.size() << std::endl;

      // END_BARRIER;
      return partitioned_octants;
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
  OrthoTree<Dim>::aid_list_t OrthoTree<Dim>::initialize_aid_list(particle_t const& particles)
    {
        LOG << "called with " << particles.getLocalNum() << " particles\n";
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

        LOG << "finished initialising the aid_list\n";
        return ret_aid_list;
    }

    template <size_t Dim>
    size_t OrthoTree<Dim>::get_num_particles_in_octant(morton_code octant)
    {
        LOG;

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

        LOG << "finished with num particles = "
            << static_cast<size_t>(upper_bound_idx - lower_bound_idx) << std::endl;
        return static_cast<size_t>(upper_bound_idx - lower_bound_idx);
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::complete_region(morton_code code_a,
                                                                morton_code code_b) {
        START_BARRIER;

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

        END_BARRIER;
        // LOG << "finished with complete_region_size = " << min_lin_tree.size() << std::endl;
        return min_lin_tree;
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::linearise_octants(
        const Kokkos::vector<morton_code>& octants) {
        START_BARRIER;

        LOG << "size: " << octants.size() << std::endl;
        Kokkos::vector<morton_code> linearised;

        for (size_t i = 0; i < octants.size() - 1; ++i) {
            if (morton_helper.is_ancestor(octants[i + 1], octants[i])) {
                continue;
            }

            linearised.push_back(octants[i]);
        }

        linearised.push_back(octants.back());

        // LOG << "finished, size is: " << linearised.size() << std::endl;

        END_BARRIER;
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
        START_BARRIER;

        LOG << "called with " << starting_octants.size() << " octants\n";  // should be 2!!
        assert(starting_octants.size() == 2 && "must receive two octants here??");
        Kokkos::vector<morton_code> T =
            complete_region(starting_octants.front(), starting_octants.back());

        LOG << "T.size() = " << T.size() << std::endl;

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

        BARRIER_LOG << "calling complete_tree" << std::endl;
        LOG << "C.size()=" << C.size() << std::endl;
        BARRIER_LOG;
        Kokkos::vector<morton_code> G = complete_tree(C);
        BARRIER_LOG << "finished complete_tree" << std::endl;
        LOG << "we now have n_octants = " << G.size() << std::endl;

        Kokkos::vector<size_t> weights = get_num_particles_in_octants_parallel(G);
        LOG << "weights have size: " << weights.size() << std::endl;
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

        BARRIER_LOG << "we survived until here :D\n";

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

        // LOG << "finished, partitioned_tree.size() = " << partitioned_tree.size() << std::endl;
        END_BARRIER;
        return partitioned_tree;
    }

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::complete_tree(
        Kokkos::vector<morton_code>& octants) {
        START_BARRIER;

        world_rank = Comm->rank();
        world_size = Comm->size();

        LOG << "called with n_octants=" << octants.size() << std::endl;
        // this removes duplicates, inefficient as of now
        std::map<morton_code, int> m;
        for ( auto octant : octants ) {
            ++m[octant];
        }

        LOG << "map has " << m.size() << "octants in total, previously we had: " << octants.size()
            << std::endl;

        octants.clear();
        for ( const auto [octant, count] : m ) {
            octants.push_back(octant);
        }

        BARRIER_LOG << "linearising octants now\n";
        LOG << " has " << octants.size() << " octants\n";
        octants = linearise_octants(octants);
        BARRIER_LOG << "done linearising octants\n";
        LOG << "linearised to octants.size()=" << octants.size() << std::endl;
        BARRIER_LOG << std::endl;

        Kokkos::vector<size_t> weights;
        for (size_t i = 0; i < octants.size(); ++i) {
            weights.push_back(1);
        }

        BARRIER_LOG << "starting partition\n";
        octants = partition(octants, weights);
        BARRIER_LOG << "finished partition\n";

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

        BARRIER_LOG << "finished send/recv\n";

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

        END_BARRIER;
        return R;
    }

    template <size_t Dim>
    Kokkos::vector<size_t> OrthoTree<Dim>::get_num_particles_in_octants_parallel(
        const Kokkos::vector<morton_code>& octants) {
        LOG;

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

        // LOG << "finished, num_particles.size() = " << num_particles.size() << std::endl;
        END_BARRIER;
        return num_particles;
    }

    template <size_t Dim>
    Kokkos::vector<size_t> OrthoTree<Dim>::get_num_particles_in_octants_seqential(
        const Kokkos::vector<morton_code>& octants) {
        LOG;
        size_t num_octs = octants.size();
        Kokkos::vector<size_t> num_particles(num_octs);
        for (size_t i = 0; i < num_octs; ++i) {
            num_particles[i] = get_num_particles_in_octant(octants[i]);
        }
        return num_particles;
    }

} // namespace ippl