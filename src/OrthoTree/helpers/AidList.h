#ifndef AID_LIST_GUARD
#define AID_LIST_GUARD

#include "OrthoTree/helpers/BoundingBox.h"
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
        AidList(size_t max_depth);

        template <size_t Dim>
        void initialize(
            const BoundingBox<Dim>& root_bounds,
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {
            if (world_rank == 0) {
                if (!is_gathered<Dim>(particles)) {
                    throw std::runtime_error("You have not gathered your particles on rank0 yet!");
                } else {
                    initialize_from_rank_0<Dim>(max_depth, root_bounds, particles);
                    logger << "Aid list is initialized with size: " << size() << endl;
                }
            }
        }

        morton_code getOctant(size_t idx) const;
        morton_code getID(size_t idx) const;
        size_t size() const;

        /**
         * @brief This should send the min octants (so front/back) for each batch for each rank
         *
         * @return std::pair<morton_code, morton_code>
         */
        std::pair<morton_code, morton_code> getMinReqOctants();

        void initializeForRank(Kokkos::View<morton_code*> octants_from_algo_4);

        size_t getNumParticlesInOctant(morton_code octant);

        size_t getNumOctantsInRange(morton_code min, morton_code max);

        // inclusive
        size_t getLowerBoundIndex(morton_code octant) const;

        // exclusive
        size_t getUpperBoundIndex(morton_code octant) const;

    private:
        template <size_t Dim>
        bool is_gathered(
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

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
        void gatherOnRank0(
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);
    };

}  // namespace ippl

#include "AidList.hpp"

#endif  // AID_LIST_GUARD