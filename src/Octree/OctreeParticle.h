/**
 * This is a class for particles to be used by the octree. It inherits from the particle base class
 * should be used with the serial layout only!
*/

#ifndef OCTREEPARTICLEGUARD
#define OCTREEPARTICLEGUARD

namespace ippl{
    
    template <class PLayout>
    class OctreeParticle : public ippl::ParticleBase< PLayout > {
    
    public:
        
        ippl::ParticleAttrib<double> rho; // charge
        
        OctreeParticle (PLayout& L) : ippl::ParticleBase<PLayout>(L) {
            this->addAttribute(rho);
        }

    private:

        

    };
}

#endif