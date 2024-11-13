#include <ostream>
#include "OrthoTree/OrthoTreeTypes.h"
#include "Communicate/Communicator.h"
#include "Communicate/Operations.h"
#include "Communicate/Request.h"
#include "Kokkos_Vector.hpp"
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
    Kokkos::vector<morton_code> OrthoTree<Dim>::partition(Kokkos::vector<morton_code>& octants, Kokkos::vector<size_t>& weights) {
      //get communicator info
      const size_t n_procs = Comm->size();
      const size_t rank = Comm->rank();

      //initialize the prefix_sum and the total weight
      Kokkos::vector<size_t> prefix_sum;
      prefix_sum.reserve(octants.size());
      size_t max = 0, total;

      //calculate the prefix_sum for the weight of each octant in octants
      // get weight here later
      for(unsigned i = 0 ; i < octants.size(); ++i){
        max += weights[i];
        prefix_sum.push_back(max);
      }



      //scan to get get the propper offsets for the prefix_sum
      Comm->scan(&max, &total, 1, std::plus<morton_code>());
      
      //calculate the actual scan prefix_sum on each processor
      for (size_t i = 0; i < prefix_sum.size(); ++i){
        prefix_sum[i] += total - max;
      }

      //broadcast the total weight to all processors
      Comm->broadcast<size_t>(&total, 1, n_procs - 1);

      //initialize the average weight
      //might want to get a double? needs checking
      size_t avg_weight = total/n_procs, k = total % n_procs;
      Kokkos::vector<morton_code> total_octants;
      Kokkos::vector<mpi::Request> requests;
      Kokkos::vector<int> sizes(n_procs);

      // initialize the start and end index for which processor receives which
      // local octants. Doing this here allows us to update these incrementally
      // in a two-pointer approach
      size_t start = 0, end = 0;
      //loop thorugh all processors
      for (size_t p = 1; p <= n_procs; ++p){

        // no need to send data to myself
        if (p -1 == rank) continue;
        //initialize the start and end index
        size_t startoffset = 0, endoffset = 0;

        // calculate the start and end offset for the processor in order
        // to distribute the remainder of total/n_procs evenly
        if (p <= k){
          startoffset = p - 1;
          endoffset = p;
        } else {
          startoffset = k;
          endoffset = k;
        }

        start = end;

        //calculate the start and end index for the processor
        for(size_t i = end; i < octants.size(); ++i){
          if(prefix_sum[i] > avg_weight * (p - 1) + startoffset){
            start = i;
            break;
          }
        }

    

        for(size_t i = start; i < octants.size(); ++i){
          if(prefix_sum[i] > avg_weight * p + endoffset){
            end = i;
            break;
          }
          if (i == octants.size() - 1)
            end = octants.size();
        }


        //if the processor is the last one, add the remaining weight
        if(p == n_procs || end > octants.size()){
          end = octants.size();
        }

        //initialize the new octants for the processor
        Kokkos::vector<morton_code> new_octants;

        //loop through the octants and add the octants to the new octants
        for(size_t i = start; i < end; ++i){
          new_octants.push_back(octants[i]);
          total_octants.push_back(octants[i]);
        }

        // send the number of new octants to the processor
        requests.push_back(mpi::Request());
        sizes[p-1] = new_octants.size();
        Comm->isend((sizes[p-1]), 1, p-1, 1, requests[requests.size()-1]);
        //send the new octants to the processor
    
        requests.push_back(mpi::Request());
        //Comm->isend(new_octants.size(), 1, p-1, 0, request1);
        if (sizes[p-1] > 0)
            Comm->isend<morton_code>(*new_octants.data(), 
                                    new_octants.size(), p-1, 
                                    0, 
                                    requests[requests.size() -1]);
      }


      std::vector<mpi::Request> receives;
      //initialize the new octants for the current processor 
      Kokkos::vector<morton_code> received_octants;
      for(size_t p = 0; p < n_procs; ++p){
        if (p == rank) continue;
        int size;
        //receives.push_back(mpi::Request());
        //receive the number of new octants
        mpi::Status stat;
        Comm->recv(&size, 1, p, 1, stat);
        //initialize the new octants
        Kokkos::vector<morton_code> octants_buffer(size);
        //receive the new octants
        //Comm->irecv(&size, 1, p, 0, receive_size);
        if (size > 0) {
          Comm->recv(octants_buffer.data(), size, p, 0, stat);
          //add the new octants to the received octants
          received_octants.insert(received_octants.end(), 
                                  octants_buffer.begin(), 
                                  octants_buffer.end());
        
        }
      }
      //add the received octants to octants and remove total_octants
      Kokkos::vector<morton_code> partitioned_octants;
      unsigned int l1 = 0, l2 = 0, l3 = 0;
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

    template <size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::complete_tree(Kokkos::vector<morton_code>& octants)
    {
        // this removes duplicates, inefficient as of now
        std::map<morton_code, int> m;
        for ( auto octant : octants ) {
            ++m[octant];
        }

        octants.clear();
        for ( const auto [octant, count] : m ) {
            octants.push_back(octant);
        }

        octants = linearise_octants(octants);

        // algo5(octants);

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

        Kokkos::vector<morton_code> R;
        // rank 0 works differently, as we need to 'simulate' push_front
        if ( world_rank == 0 ) {
            R.push_back(first_rank0);
            for ( morton_code elem : algo2(first_rank0, octants[0]) ) {
                R.push_back(elem);
            }
        }

        const size_t n = octants.size();
        for ( size_t i = 0; i < n - 1; ++i ) {
            for ( morton_code elem : algo2(octants[i], octants[i+1]) ) {
                R.push_back(elem);
            }
            R.push_back(octants[i]);
        }

        if ( world_rank == world_size - 1 ) {
            R.push_back(octants[n - 1]);
        }

        return R;
    }

    template<size_t Dim>
    Kokkos::vector<size_t> OrthoTree<Dim>::get_num_particles_in_octants_parallel(const Kokkos::vector<morton_code>& octants)
    {

        //get communicator info
        const size_t n_procs = Comm->size();
        const size_t rank = Comm->rank();

        mpi::Status stat;

        if(rank == 0){
          for(size_t i = 1; i < n_procs; ++i){

            int req_size;
            Comm->recv(req_size,1,i,1,stat);
            Kokkos::vector<morton_code> octants_buffer(req_size);
            Kokkos::vector<size_t> num_particles;
            Comm->recv(octants_buffer.data(), req_size, i, 0, stat);

            num_particles = get_num_particles_in_octants_seqential(octants_buffer);
            Comm->send(*num_particles.data(), req_size, i, 0);
          }

          Kokkos::vector<size_t> num_particles = get_num_particles_in_octants_seqential(octants);
          return num_particles;


        }

        else{
          int req_size = octants.size();
          Comm->send(req_size, 1, 0, 1);
          Comm->send(*octants.data(), octants.size(), 0, 0);
          Kokkos::vector<size_t> num_particles(req_size);
          Comm->recv(num_particles.data(), req_size, 0, 0, stat);
          return num_particles;

        }

    }

    template<size_t Dim>
    Kokkos::vector<morton_code> OrthoTree<Dim>::get_num_particles_in_octants_seqential(const Kokkos::vector<morton_code>& octants)
    {
        size_t num_octs = octants.size();
        Kokkos::vector<size_t> num_particles(num_octs);
        for (size_t i = 0; i < num_octs; ++i) {
            num_particles[i] = get_num_particles_in_octant(octants[i]);
        }
        return num_particles;
    }
  
} // namespace ippl