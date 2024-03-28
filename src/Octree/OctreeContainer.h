/**
 * OctreeContainer class for IPPL
 * It is based on the freely available implementation : https://github.com/attcs/Octree
*/

#ifndef ORTHOTREECONTAINERGUARD
#define ORTHOTREECONTAINERGUARD

#include "Octree/OctreeParticle.h"

namespace ippl {

    namespace Octree{

        /**
         * @class OctreeContainer: Tree container class
         * @tparam Octree: The type of Orthotree, for now this is just a balanced Octree
         * @tparam PLayout: The IPPL particle layout, containing the view of positions and charge attributes
        */
        template<typename Octree, class PLayout=ippl::ParticleSpatialLayout<double, 3>>
        class OctreeContainer{
        
        public: // Typenames

            using AD                    = typename Octree::AD;
            using vector_type           = typename Octree::vector_type;
            using box_type              = typename Octree::box_type;
            using max_element_type      = typename Octree::max_element_type;
            using geometry_type         = typename Octree::geometry_type;
        
        protected: // Member variables

            Octree tree_m;
            OctreeParticle<PLayout> particles_m;
            vector<vector_type const> vpt={};

        public: // Constructors

            OctreeContainer() noexcept = default;

            OctreeContainer(
                OctreeParticle<PLayout>&        particles,
                depth_type                      nDepthMax=0,
                std::optional<box_type> const&  oBoxSpace = std::nullopt, 
                max_element_type                nElementMaxInNode = Octree::max_element_default) noexcept
                {
                    particles_m.initialize(particles.getLayout());
                    particles_m.create(particles.getLocalNum());
                    //particles_m.R = particles.R;
                    //particles_m.rho = particles.rho;
                    //particles.tindex = particles.tindex;
                                        
                    std::cout <<  particles.getLocalNum() << std::endl;    

                    for(unsigned int i=0; i<particles.getLocalNum(); ++i){
                        vpt.push_back(particles.R(i));
                        std::cout << vpt[i][0] << " " << vpt[i][1] << " " << vpt[i][2] << std::endl;
                        //particles_m.R(i) = particles.R(i);
                        //particles_m.rho(i) = particles.rho(i);
                    }
                    
                    std::cout << "here" << std::endl;
                    
                    //tree_m = Octree(vpt, nDepthMax, oBoxSpace, nElementMaxInNode);
                    Octree::Create(tree_m, vpt, nDepthMax, oBoxSpace, nElementMaxInNode);
                    //tree_m.BalanceOctree(vpt);
                    
                }
            
            
        public: // Member functions

            constexpr Octree const& GetCore() const noexcept {return tree_m;}
            constexpr OctreeParticle<PLayout> const& GetData() const noexcept {return particles_m;}
        };

    } // namespace Orthotree

} // namespace ippl

#endif