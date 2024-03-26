/**
 * OctreeContainer class for IPPL
 * It is based on the freely available implementation : https://github.com/attcs/Octree
*/

#ifndef ORTHOTREECONTAINERGUARD
#define ORTHOTREECONTAINERGUARD

#include "Octree/OctreeParticle.h"

namespace ippl {

    namespace OrthoTree{

        /**
         * @class OrthoTreeContainer: Tree container class
         * @tparam OrthoTree: The type of Orthotree, for now this is just a balanced Octree
         * @tparam PLayout: The IPPL particle layout, containing the view of positions and charge attributes
        */
        template<typename OrthoTree, class PLayout=ippl::ParticleSpatialLayout<double, 3>>
        class OrthoTreeContainer{
        
        public:
            using AD = typename OrthoTree::AD;
            using vector_type = typename OrthoTree::vector_type;
            using box_type = typename OrthoTree::box_type;
            using max_element_type = typename OrthoTree::max_element_type;
            using geometry_type = typename OrthoTree::geometry_type;
        
        protected:

            OrthoTree tree_m;
            OctreeParticle<PLayout> particles_m;

        public: // Constructors

            OrthoTreeContainer() noexcept = default;

            OrthoTreeContainer( OctreeParticle<PLayout const> const& particles,
                                depth_type nDepthMax = 0, 
                                std::optional<box_type> const& oBoxSpace = std::nullopt, 
                                max_element_type nElementMaxInNode = OrthoTree::max_element_default/*, 
                                bool fParallelCreate = false*/) noexcept       
                : particles_m(particles)     
            {
                vector<ippl::Vector<double,3>> points={};
                for(unsigned int i=0; i<particles_m.getLocalNum(); ++i){
                    points.push_back(particles_m.R(i));
                }
                OrthoTree::Create(tree_m, points, nDepthMax, oBoxSpace, nElementMaxInNode);
                tree_m.BalanceOctree(points);
            }
            
        public: // Member function

            constexpr OrthoTree const& GetCore() const noexcept {return tree_m;}
            constexpr OctreeParticle<PLayout> const& GetData() const noexcept {return particles_m;}
        };

    } // namespace Orthotree

} // namespace ippl

#endif