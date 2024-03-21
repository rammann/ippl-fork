/**
 * Octree class for IPPL
 * It is based on the freely available implementation : https://github.com/attcs/Octree
*/
#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include <span>
#include <queue>
#include <Kokkos_Vector.hpp>
#include "OctreeAdaptors.h"

namespace ippl
{
namespace OrthoTree
{
    
    /**
     * @class OrthoTreeBase: Base class for the tree
    */
    template<dim_type nDimension, typename vector_type_, typename box_type_, typename adaptor_type = AdaptorGeneral<nDimension, vector_type_, box_type_, double>, typename geometry_type_ = double>
    class OrthoTreeBase
    {
        static_assert(nDimension == 3, "implementation limited to dim = 3 case");

    public: // typedefs

        static autoce IsLinearTree = nDimension < 15; // Requiredment for Morton ordering

        using child_id_type = uint64_t; 
        using morton_grid_id_type = uint32_t; // increase if more depth / higher dimension is desired
        using morton_node_id_type = morton_grid_id_type;
        using max_element_type = uint32_t;
        using geometry_type = geometry_type_;
        using vector_type = vector_type_;
        using box_type = box_type_;
        using AD = adaptor_type;

    protected:
        // Max children per node
        static autoce m_nChild = pow_ce(2, nDimension); 

        // Type system determined maximal depth.
        static autoce m_nDepthMaxTheoretical = depth_type((CHAR_BIT * sizeof(morton_node_id_type) - 1/*sentinal bit*/) / nDimension);

    public:
        /**
         * @class Node
         * @param children_m vector of morton ids corresponding to children nodes
         * @param vid_m If this node is a leaf, this contains the ids of the points contained
         * @param box
        */
        class Node
        {
        private:

            Kokkos::vector<morton_node_id_type> children_m;
        
        public: 

            Kokkos::vector<entity_id_type> vid_m;
            box_type box = {};
            morton_node_id_type parent_m;
        };
    };



} // namespace OrthoTree
} // namespace ippl

#include "OctreeContainer.h"

#ifdef undef_autoc
#undef autoc
#undef undef_autoc
#endif

#ifdef undef_autoce
#undef autoce
#undef undef_autoce
#endif

#ifdef UNDEF_HASSERT
#undef HASSERT
#undef UNDEF_HASSERT
#endif

#undef LOOPIVDEP


#endif