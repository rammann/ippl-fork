#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include <Kokkos_Vector.hpp>
#include <Kokkos_UnorderedMap.hpp>
#include <Kokkos_Pair.hpp>
#include <Kokkos_StdAlgorithms.hpp>



namespace ippl
{

// Types

struct BoundingBox{
    ippl::Vector<double, 3> Min;
    ippl::Vector<double, 3> Max;
}; 

using dim_type                  = unsigned short int;

// Node Types
using morton_node_id_type       = unsigned int;
using grid_id_type              = unsigned int;

// Particle Types
using particle_type             = OctreeParticle<ippl::ParticleSpatialLayout<double,3>>;
using entity_id_type            = size_t;
using position_type             = ippl::Vector<double,3>;

using box_type                  = BoundingBox;
using depth_type                = unsigned int;



class OrthoTreeNode
{
public: // Member Variables
    
    Kokkos::vector<morton_node_id_type> children_m;
    Kokkos::vector<morton_node_id_type> vid_m;
    box_type                            boundingbox_m;
    morton_node_id_type                 parent_m;

public: // Member Functions

    void AddChildren(morton_node_id_type kChild){
        children_m.push_back(kChild);
    }

    void AddChildInOrder(morton_node_id_type kChild){
        morton_node_id_type idx = children_m.lower_bound(0,children_m.size(), kChild);
        if(idx != children_m.size() && children_m[idx] == kChild) return;
        if(idx == children_m.size()-1){
            children_m.insert(children_m.begin()+idx+1, kChild);
            return;
        } 
        children_m.insert(children_m.begin()+idx, kChild);
        
    }

    bool HasChild(morton_node_id_type kChild){
        auto it = children_m.find(kChild);
        if(it == children_m.end()) return false;
        else return true;
    }

    bool IsAnyChildExist(){
        return !children_m.empty();
    }

    Kokkos::vector<morton_node_id_type>& GetChildren(){
        return children_m;
    }

};

using container_type = Kokkos::UnorderedMap<morton_node_id_type, OrthoTreeNode>;

class OrthoTree
{
private:
    
    container_type                  nodes_m;        // Kokkos::UnorderedMap of {morton_node_id, Node} pairs
    box_type                        box_m;          // Bounding Box of the Tree = Root node's box
    depth_type                      maxdepth_m;     // Max Depth of Tree
    size_t                          maxelements_m;  // Max points per node
    grid_id_type                    rasterresmax_m; // Max boxes per dim
    ippl::Vector<double,3>          rasterizer_m;   // Describes how many nodes make up 1 unit of length per dim
    const dim_type                  dim_m = 3;      // Dimension (fixed at 3 for now)

public: // Constructors

    OrthoTree () = default;
    OrthoTree (particle_type const& particles, depth_type MaxDepth, box_type Box, size_t MaxElements)
    {
        this->box_m             = Box;
        this->maxdepth_m        = MaxDepth;
        this->maxelements_m     = MaxElements;
        this->rasterresmax_m    = Kokkos::exp2(MaxDepth);
        this->rasterizer_m      = GetRasterizer(Box, this->rasterresmax_m);

        const size_t n = particles.getLocalNum(); // use getGlobalNum() instead? 

        nodes_m = container_type(EstimateNodeNumber(n, MaxDepth, MaxElements));

        // Root (key, node) pair
        morton_node_id_type kRoot = 1;
        OrthoTreeNode NodeRoot;
        NodeRoot.boundingbox_m = Box;
        nodes_m.insert(kRoot, NodeRoot);

        // Vector of point ids
        Kokkos::vector<entity_id_type> vidPoint(n);
        for(unsigned i=0; i<n; ++i) vidPoint[i] = i;

        // Vector of corresponding poisitions
        Kokkos::vector<position_type> positions(n);
        for(unsigned i=0; i<n; ++i) positions[i] = particles.R(i);
 
        // Vector of aid locations
        Kokkos::vector<Kokkos::pair<entity_id_type, morton_node_id_type>> aidLocations(n);

        // transformation of (id, position(id)) -> (id, aidlocation(id))
        std::transform( positions.begin(), positions.end(), vidPoint.begin(), aidLocations.begin(),
        [=](position_type pt, entity_id_type id) -> Kokkos::pair<entity_id_type, morton_node_id_type>
        {
            return {id, this->GetLocationId(pt)};
        });
    }

public: // Aid Functions

    unsigned int EstimateNodeNumber(unsigned int nParticles, depth_type MaxDepth, unsigned int nMaxElements){
        
        if (nParticles < 10) return 10;
        
        const double rMult = 1.5;

        
        // for smaller problem size
        if ((MaxDepth + 1) * dim_m < 64){
            size_t nMaxChild = size_t{ 1 } << (MaxDepth * dim_m);
            auto const nElementsInNode = nParticles / nMaxChild;
            if (nElementsInNode > nMaxElements / 2) return nMaxChild;
        }

        // for larget problem size
        auto const nElementInNodeAvg = static_cast<float>(nParticles) / static_cast<float>(nMaxElements);
        auto const nDepthEstimated = std::min(MaxDepth, static_cast<depth_type>(std::ceil((log2f(nElementInNodeAvg) + 1.0) / static_cast<float>(dim_m))));
        if (nDepthEstimated * dim_m < 64) return static_cast<size_t>(rMult * (1 << nDepthEstimated * dim_m));
        return static_cast<size_t>(rMult * nElementInNodeAvg);
    }

    ippl::Vector<double,3> GetRasterizer(box_type Box, grid_id_type nDivide){
        const double ndiv = static_cast<double>(nDivide);
        ippl::Vector<double,3> rasterizer;
        for(dim_type i=0; i<dim_m; ++i){
            double boxsize = Box.Max[i] - Box.Min[i];
            rasterizer[i] = boxsize == 0 ? 1.0 : (ndiv/boxsize);
        }
        std::cout << rasterizer << std::endl;
        return rasterizer;
    }

    morton_node_id_type GetLocationId(position_type pt){
        return MortonEncode(GetGridId(pt));
    }

    ippl::Vector<grid_id_type,3> GetGridId(position_type pt){
        ippl::Vector<grid_id_type,3> aid;
        for(dim_type i=0; i<dim_m; ++i){
            double r_i = pt[i] - box_m.Min[i];
            double raster_id = r_i * rasterizer_m[i];
            aid[i] = static_cast<grid_id_type>(raster_id);
        }
        return aid;
    }

    // Only works for dim = 3 for now
    morton_node_id_type MortonEncode(ippl::Vector<grid_id_type,3> aidGrid){
        assert(dim_m == 3);
        return (part1By2(aidGrid[2]) << 2) + (part1By2(aidGrid[1]) << 1) + part1By2(aidGrid[0]);
    }

    static constexpr morton_node_id_type part1By2(grid_id_type n) noexcept
    {
        // n = ----------------------9876543210 : Bits initially
        // n = ------98----------------76543210 : After (1)
        // n = ------98--------7654--------3210 : After (2)
        // n = ------98----76----54----32----10 : After (3)
        // n = ----9--8--7--6--5--4--3--2--1--0 : After (4)
        n = (n ^ (n << 16)) & 0xff0000ff; // (1)
        n = (n ^ (n << 8)) & 0x0300f00f; // (2)
        n = (n ^ (n << 4)) & 0x030c30c3; // (3)
        n = (n ^ (n << 2)) & 0x09249249; // (4)
        return static_cast<morton_node_id_type>(n);
    }


};











}
#endif