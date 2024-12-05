#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include "OrthoTree/helpers/MortonHelper.h"

namespace ippl {

    /**
     * @brief Baisc idea of a better AidList
     *
     */
    class AidList {
        const size_t world_rank;
        const size_t world_size;
        const size_t max_depth;

        Inform logger;

        Kokkos::View<size_t* [2]> aid_list_m;  // aid_list_m = [{morton_code, particle_id}]

    public:
        template <size_t Dim>
        AidList(size_t max_depth, const BoundingBox<Dim>& root_bounds,
                OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles)
            : world_rank(Comm->rank())
            , world_size(Comm->size())
            , max_depth(max_depth)
            , logger("AidList", std::cout, INFORM_ALL_NODES) {
            // TODO: check if gathered or not, if not gather on rank 0!

            logger.setOutputLevel(5);
            logger.setPrintNode(INFORM_ALL_NODES);

            initialize_from_rank_0(max_depth, root_bounds, particles);
            logger << "Initialized AidList" << endl;
        }

        /**
         * @brief This should send the min octants (so front/back) for each batch for each rank
         *
         * @return std::pair<morton_code, morton_code>
         */
        std::pair<morton_code, morton_code> getMinReqOctants();

        void initializeForRank(Kokkos::view<morton_code*> octants_from_algo_4);

        size_t getNumParticlesInOctant(morton_code octant);

        size_t getNumOctantsInRange(morton_code min, morton_code max);

    private:
        template <size_t Dim>
        bool is_gathered(
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
            if (world_rank != 0) {
                throw std::exception("only call AidList::is_gathered from rank 0!");
            }

            return particles.getLocalNum() == particles.getTotalNum();
        }

        /**
         * @brief This function excpects all particles to be on rank 0 and generates the
         * corresponding aid list.
         *
         * @tparam Dim
         * @param particles
         */
        template <size_t Dim>
        void initialize_from_rank_0(
            size_t max_depth, const BoundingBox<Dim>& root_bounds,
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        /**
         * @brief If the AidList gets initialized with a scattered list of particles (each proc
         * has certain number) we first call this function to collect all particles on rank0
         *
         * @tparam Dim
         * @param particles
         */
        template <size_t Dim>
        void collect(OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);
    };

    template <size_t Dim>
    void AidList::initialize_from_rank_0(
        size_t max_depth, const BoundingBox<Dim>& root_bounds,
        OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
        using real_coordinate = real_coordinate_template<Dim>;

        if (world_rank != 0) {
            return;
        }

        if (particles.getLocalNum() != particles.getTotalNum()) {
            throw std::runtime_error(
                "can only initialize if all particles are gathered on rank 0!");
        }

        const size_t n_particles = particles.getLocalNum();
        const size_t grid_size   = (size_t(1) << max_depth_m);

        const real_coordinate root_bounds_size = root_bounds_m.get_max() - root_bounds_m.get_min();
        this->aid_list_m = Kokkos::View<morton_code* [2]>("aid_list_m", n_particles);

        for (size_t i = 0; i < n_particles; ++i) {
            // normalize particle coordinate inside the grid
            const real_coordinate normalized =
                (particles.R(i) - root_bounds_m.get_min()) / root_bounds_size;
            // calculate the grid coordinate relative to the bounding box and grid size
            const grid_coordinate grid_coord = static_cast<grid_coordinate>(normalized * grid_size);

            aid_list_m[i][0] = morton_helper.encode(grid_coord, max_depth_m);
            aid_list_m[i][1] = i;
        }

        // TODO: sort the view
    }

    std::pair<morton_code, morton_code> AidList::getMinReqOctants() {
        morton_code min, max;
        if (world_rank == 0) {
            const size_t size        = aid_list_m.size();
            const size_t batch_size  = size / world_size;
            const size_t rank_0_size = batch_size + (size % batch_size);

            for (size_t target_rank = 1; target_rank < world_size; ++target_rank) {
                const size_t start = ((target_rank - 1) * batch_size) + rank_0_size;
                const size_t end   = start + batch_size;

                min = aid_list_m[start][0];
                max = aid_list_m[end - 1][0];

                Comm->send(min, 1, target_rank, 0);
                Comm->send(max, 1, target_rank, 0);

                logger << "sent min: " << std::hex << min << ", max: " << max << std::dec
                       << " to rank " << target_rank << endl;
            }

            min = aid_list_m[0][0];
            max = aid_list_m[rank_0_size - 1][0];
        } else {
            mpi::Status stat1, stat2;
            Comm->recv(&min, 1, 0, 0, stat1);
            Comm->recv(&max, 1, 0, 0, stat2);

            logger << "received min: " << std::hex << min << ", max: " << max << std::dec << endl;
        }

        return std::make_pair<morton_code, morton_code>(min, max);
    }

}  // namespace ippl

#endif  // ORTHOTREE_GUARD