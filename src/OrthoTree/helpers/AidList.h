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

        /**
         * As of now we have two seperate views to store octants and particle ids.
         * They will only be accessed through the helper functions below, so if we decide to change
         * implementation we (should) only have to change those and everything sould still work.
         */
        Kokkos::View<morton_code*> octants;
        Kokkos::View<size_t*> particle_ids;

        /*
        TODO:
1.
        Kokkok flattens views, so as of now we have: view={[morton_codes], [id's]}, this sucks for
        cache locality...
        -> use a 1d view where we alternate octants and ids.

2.
        Experiment around to see if this actually works.

3.
        If it works, start writing tests for this class

3.5
        Rewrite the AidList s.t. it works with arbitrary particle types, not just with the OrthoTree
particles:)

4.
        Merge into dphpc octree and update the AidList type in the OrthoTree.h file.


What we need:
        1. efficient and safe AidList impl.
        2. methods that are easy to use with:
            - get num particles in octant (local on rank)
            - get num particles in octant global (will call upon rank 0)
            - get relevant min/max octants
            - automatically gather particles on rank0 if its not done so already
        3.

        */

    public:
        AidList(size_t max_depth);

        template <size_t Dim>
        void initialize(
            const BoundingBox<Dim>& root_bounds,
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        size_t size() const { return octants.extent(0); }

        morton_code getOctant(size_t idx) const { return octants(idx); }
        size_t getID(size_t idx) const { return particle_ids(idx); }

        void setOctant(morton_code octant, size_t idx) { octants(idx) = octant; }
        void setID(size_t particle_id, size_t idx) { particle_ids(idx) = particle_id; }

        auto getOctants() const { return Kokkos::subview(octants, Kokkos::ALL()); }
        auto getParticleIDs() const { return Kokkos::subview(particle_ids, Kokkos::ALL()); }

        auto& getOctants() { return octants; }
        auto& getParticleIDs() { return particle_ids; }

        void resize(size_t num_elements) {
            Kokkos::resize(octants, num_elements);
            Kokkos::resize(particle_ids, num_elements);
        }

        /**
         * @brief This should send the min octants (so front/back) for each batch for each rank
         *
         * @return std::pair<morton_code, morton_code>
         */
        std::pair<morton_code, morton_code> getMinReqOctants();

        /**
         * @brief Returns the number of particles in the given octant.
         * This function assumes that the AidList is initialized.
         */
        size_t getNumParticlesInOctant(morton_code octant) const;

        /**
         * @brief Returns the lowest index s.t. octants(index-1) < octant <= octants(index)
         * INCLUSIVE
         */
        size_t getLowerBoundIndex(morton_code octant) const;

        /**
         * @brief Returns the highest index s.t. octants(index-1) <= octant < octants(index)
         */
        size_t getUpperBoundIndexExclusive(morton_code octant) const;

        /**
         * @brief Returns the highest index s.t. octants(index-1) <= octant <= octants(index)
         */
        size_t getUpperBoundIndexInclusive(morton_code octant) const;

    private:
        /**
         * @brief Checks if the particles have been gathered on the rank it is called on.
         * Will throw if they are not gathered.
         */
        template <size_t Dim>
        bool is_gathered(
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        /**
         * @brief Initialises the AidList with {morton_code, particle_id} pairs.
         * This funciton assumes that all particles have been gathered on the given rank, will throw
         * if they are not.
         */
        template <size_t Dim>
        void initialize_from_rank(
            size_t max_depth, const BoundingBox<Dim>& root_bounds,
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);

        /**
         * @brief Gathers the particles on the rank that calls this function.
         */
        template <size_t Dim>
        void gatherOnRank(
            OrthoTreeParticle<ippl::ParticleSpatialLayout<double, Dim>> const& particles);
    };

}  // namespace ippl

#include "AidList.hpp"

#endif  // AID_LIST_GUARD