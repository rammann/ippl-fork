/**
 * This is a class for particles to be used by the octree. It inherits from the particle base class
 * should be used with the serial layout only!
*/

#ifndef OCTREEPARTICLEGUARD
#define OCTREEPARTICLEGUARD

namespace ippl{
    
    /**
     * @class OctreeParticle: Class that implements 3D particles with rho attribute 
     * @tparam PLayout: Particle Layout type (for this case ippl::ParticleSpatialLayout<double, 3>)
     * 
     * The points can be accessed through this->R(index), which return the i-th point as a ippl::Vector<double, 3>.
     * rho(i) returns the charge of the i-th particle
    */
    template <class PLayout = ippl::ParticleSpatialLayout<double, 3> >
    class OctreeParticle : public ippl::ParticleBase<PLayout> {
    
    public:

        unsigned int tindex; // index of first targetpoint
        
        ippl::ParticleAttrib<double> rho; // charge

        OctreeParticle (PLayout& L, unsigned int i) : ippl::ParticleBase<PLayout>(L), tindex(i) {
            this->addAttribute(rho);
        }

    };
}

#endif