#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include "MortonHelper.h"

namespace ippl {

    /**
     * @brief Baisc idea of a better AidList
     *
     */
    struct AidList {
        size_t world_rank;
        size_t world_size;
        Kokkos::View<size_t* [2]> aid_list_m;  // aid_list_m = [{morton_code, particle_id}]

        template <size_t Dim>
        AidList(OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles)
            : world_rank(Comm->rank())
            , world_size(Comm->size()) {
            if (world_rank == 0) {
                aid_list_m = Kokkos::View<morton_code* [2]>("aid_list_m", particles.getTotalNum());
            }

            // TODO: check if gathered or not, if not gather on rank 0!

            initialize_from_rank_0(particles);
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
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        /**
         * @brief If the AidList gets initialized with a scattered list of particles (each proc has
         * certain number) we first call this function to collect all particles on rank0
         *
         * @tparam Dim
         * @param particles
         */
        template <size_t Dim>
        void collect(OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);
    };

    template <size_t Dim>
    AidList::AidList(OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles) {

    }

}  // namespace ippl

#endif  // ORTHOTREE_GUARD