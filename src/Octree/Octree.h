/**
 * Octree class for IPPL
 * It is based on the freely available implementation : https://github.com/attcs/Octree
*/
#ifndef ORTHOTREE_GUARD
#define ORTHOTREE_GUARD

#include <span>
#include <queue>
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

        static autoce is_linear_tree = nDimension < 15; // Requiredment for Morton ordering

        using child_id_type = uint64_t; 
        using morton_grid_id_type = uint32_t; // increase if more depth / higher dimension is desired
        using morton_node_id_type = morton_grid_id_type;
        using morton_node_id_type_cref = morton_node_id_type;
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
         * @param box extent of node
        */
        class Node
        {
        private:

            vector<morton_node_id_type> m_children; // vector of morton ids of children
        
        public: 

            vector<entity_id_type> m_vid = {}; // contains ids of points if leaf node
            box_type box = {}; // extent of node
            morton_node_id_type m_parent; // morton id of parent node
        
        public: 
            
            constexpr void AddChild(morton_node_id_type_cref kChild) noexcept { 
                m_children.emplace_back(kChild); 
            }
            
            constexpr void AddChildInOrder(morton_node_id_type_cref kChild) noexcept {
                auto it = std::end(m_children);
                it = std::lower_bound(m_children.begin(), m_children.end(), kChild);
                if (it != m_children.end() && *it == kChild) return;
                m_children.insert(it, kChild);
            }

            constexpr bool HasChild(morton_node_id_type_cref kChild) const  noexcept {
                return std::ranges::binary_search(m_children, kChild);
            }

            constexpr bool IsAnyChildExist() const noexcept {
                 return !m_children.empty(); 
            }
            
            constexpr vector<morton_node_id_type> const& GetChildren() const noexcept {
                 return m_children; 
            }
        }; // class Node
    
    protected: // Member Variables

        // Container type for linear tree (< 15 dims)
        template<typename data_type>
        using container_type = unordered_map<morton_node_id_type, data_type>;

        // Member Variables
        container_type<Node> m_nodes;
        box_type m_box = {};
        depth_type m_nDepthMax = {};
        grid_id_type m_nRasterResolutionMax = {};
        grid_id_type m_idSlotMax = {};
        max_element_type m_nElementMax = 11;
        double m_rVolume = {};
        array<double, nDimension> m_aRasterizer;
        array<double, nDimension> m_aBoxSize;
        array<double, nDimension> m_aMinPoint;

    protected: // Aid functions

        template<size_t N>
        static inline child_id_type convertMortonIdToChildId(bitset_arithmetic<N> const& bs) noexcept{
            assert(bs <= bitset_arithmetic<N>(std::numeric_limits<size_t>::max()));
            return bs.to_ullong();
        }
        
        static constexpr child_id_type convertMortonIdToChildId(uint64_t morton) noexcept { 
            return morton; 
        }

        // creates and n long vector for entity ids
        static constexpr vector<entity_id_type> generatePointId(size_t n) noexcept {
            auto vidPoint = vector<entity_id_type>(n);
            std::iota(std::begin(vidPoint), std::end(vidPoint), 0);
            return vidPoint;
        }
    
    protected: // Grid functions

        // returns how many segments make up 1 unit length in each dim
        static constexpr std::tuple<array<double, nDimension>, array<double, nDimension>> getGridRasterizer(vector_type const& p0, vector_type const& p1, grid_id_type n_divide) noexcept{
            auto ret = std::tuple<array<double, nDimension>, array<double, nDimension>>{};
            auto& [aRasterizer, aBoxSize] = ret;
            autoc rn_divide = static_cast<double>(n_divide);
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension){
                aBoxSize[iDimension] = static_cast<double>(adaptor_type::point_comp_c(p1, iDimension) - adaptor_type::point_comp_c(p0, iDimension));
                aRasterizer[iDimension] = aBoxSize[iDimension] == 0 ? 1.0 : (rn_divide / aBoxSize[iDimension]);
            }

            return ret;
        }

        // returns grid-id (on finest level) of point "pe" relative to min(m_box)
        constexpr array<grid_id_type, nDimension> getGridIdPoint(vector_type const& pe) const noexcept
        {
            auto aid = array<grid_id_type, nDimension>{};
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension){
                autoc local_comp = adaptor_type::point_comp_c(pe, iDimension) - adaptor_type::point_comp_c(adaptor_type::box_min_c(this->m_box), iDimension);
                auto raster_id = static_cast<double>(local_comp) * this->m_aRasterizer[iDimension];
                aid[iDimension] = std::min<grid_id_type>(this->m_idSlotMax, static_cast<grid_id_type>(raster_id));
            }
            return aid;
        }

        // returns grid-id (on finest level) of point "pe" relative to 0
        constexpr array<grid_id_type, nDimension> getRelativeGridIdPoint(vector_type const& pe_r) const noexcept
        {
            auto aid = array<grid_id_type, nDimension>{};
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
            {
                autoc local_comp = adaptor_type::point_comp_c(pe_r, iDimension);
                auto raster_id = static_cast<double>(local_comp) * this->m_aRasterizer[iDimension];
                aid[iDimension] = std::min<grid_id_type>(this->m_idSlotMax, static_cast<grid_id_type>(raster_id));
            }
            return aid;
        }

        inline Node& createChild(Node& nodeParent, child_id_type iChild, morton_node_id_type_cref kChild)noexcept
        {
            assert(iChild < this->m_nChild);
            if(!nodeParent.HasChild(kChild)){
                nodeParent.AddChildInOrder(kChild);
            }
            
            auto& nodeChild = m_nodes[kChild]; 

            if constexpr (std::is_integral_v<geometry_type>){
                std::array<double, nDimension> ptNodeMin = this->m_aMinPoint, ptNodeMax;

                autoc nDepth = this->GetDepth(kChild);
                auto mask = morton_node_id_type{ 1 } << (nDepth * nDimension - 1);

                auto rScale = 1.0;
                for (depth_type iDepth = 0; iDepth < nDepth; ++iDepth){
                    rScale *= 0.5;
                    for (dim_type iDimension = nDimension; iDimension > 0; --iDimension){
                        bool const isGreater = (kChild & mask);
                        ptNodeMin[iDimension - 1] += isGreater * this->m_aBoxSize[iDimension - 1] * rScale;
                        mask >>= 1;
                    }
                }

                LOOPIVDEP
                for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension){
                    ptNodeMax[iDimension] = ptNodeMin[iDimension] + this->m_aBoxSize[iDimension] * rScale;
                }
                
                LOOPIVDEP
                for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension){
                    AD::point_comp(AD::box_min(nodeChild.box), iDimension) = static_cast<geometry_type>(ptNodeMin[iDimension]);
                    AD::point_comp(AD::box_max(nodeChild.box), iDimension) = static_cast<geometry_type>(ptNodeMax[iDimension]);
                }
            }
            else
            {
                LOOPIVDEP
                for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                {
                    autoc fGreater = ((child_id_type{ 1 } << iDimension) & iChild) > 0;
                    AD::point_comp(AD::box_min(nodeChild.box), iDimension) =
                        fGreater * (AD::point_comp_c(AD::box_max_c(nodeParent.box), iDimension) + AD::point_comp_c(AD::box_min_c(nodeParent.box), iDimension)) * geometry_type{ 0.5 } +
                        (!fGreater) * AD::point_comp_c(AD::box_min_c(nodeParent.box), iDimension);

                    AD::point_comp(AD::box_max(nodeChild.box), iDimension) =
                        fGreater * AD::point_comp_c(AD::box_max_c(nodeParent.box), iDimension) +
                        (!fGreater) * ((AD::point_comp_c(AD::box_max_c(nodeParent.box), iDimension) + AD::point_comp_c(AD::box_min_c(nodeParent.box), iDimension)) * geometry_type{ 0.5 });
                }
            }
            return nodeChild;
        }

        constexpr morton_grid_id_type getLocationId(vector_type const& pt) const noexcept {
            return MortonEncode(this->getGridIdPoint(pt));
        }

        bool isEveryItemIdUnique() const noexcept
        {
            auto ids = vector<entity_id_type>();
            ids.reserve(100);
            std::ranges::for_each(m_nodes, [&](auto& node)
            {
                ids.insert(end(ids), begin(node.second.vid), end(node.second.vid));
            });

            std::ranges::sort(ids);
            autoc itEndUnique = std::unique(begin(ids), end(ids));
            return itEndUnique == end(ids);
        }
        
        template<bool bCheckUniqness>
        bool insert(morton_node_id_type_cref kNode, morton_node_id_type_cref kNodeSmallest, entity_id_type id, bool fInsertToLeaf) noexcept
        {
            if (kNode == kNodeSmallest)
            {
                cont_at(this->m_nodes, kNode).vid.emplace_back(id);
                if constexpr (bCheckUniqness)
                assert(this->isEveryItemIdUnique()); // Assert means: index is already added. Wrong input!
                return true;
            }

            if (fInsertToLeaf)
            {
                auto& nodeNew = this->m_nodes[kNode];
                nodeNew.vid.emplace_back(id);
                nodeNew.box = this->CalculateExtent(kNode);

                // Create all child between the new (kNode) and the smallest existing one (kNodeSmallest)
                auto kNodeParent = kNode;
                do
                {
                auto kNodeChild = kNodeParent;
                kNodeParent >>= nDimension;
                assert(IsValidKey(kNodeParent));
                auto& nodeParent = this->m_nodes[kNodeParent];
                nodeParent.AddChildInOrder(kNodeChild);
                nodeParent.box = this->CalculateExtent(kNodeParent);
                } while (kNodeParent != kNodeSmallest);
            }
            else
            {
                autoc itNode = this->m_nodes.find(kNodeSmallest);
                if (itNode->second.IsAnyChildExist())
                {
                autoc nDepth = this->GetDepth(kNodeSmallest);
                autoc kNodeChild = kNode << (nDimension * (this->m_nDepthMax - nDepth - 1));
                autoc iChild = getChildPartOfLocation(kNodeChild);
                auto& nodeChild = this->createChild(itNode->second, iChild, kNodeChild);
                nodeChild.vid.emplace_back(id);
                }
                else
                itNode->second.vid.emplace_back(id);
            }

            if constexpr (bCheckUniqness)
                assert(this->isEveryItemIdUnique()); // Assert means: index is already added. Wrong input!

            return true;
        }

        template<typename data_type = Node>
        static void reserveContainer(map<morton_node_id_type, data_type, bitset_arithmetic_compare>&, size_t) noexcept {};
    
        template<typename data_type = Node>
        static void reserveContainer(unordered_map<morton_node_id_type, data_type>& m, size_t n) noexcept { m.reserve(n); };
        
    public: // Static aid functions

        static constexpr size_t EstimateNodeNumber(size_t nElement, depth_type nDepthMax, max_element_type nElementMax) noexcept{
            assert(nElementMax > 0);
            assert(nDepthMax > 0);
            
            if (nElement < 10)
                return 10;
            
            autoce rMult = 1.5;
            if ((nDepthMax + 1) * nDimension < 64)
            {
                size_t const nMaxChild = size_t{ 1 } << (nDepthMax * nDimension);
                autoc nElementInNode = nElement / nMaxChild;
                if (nElementInNode > nElementMax / 2)
                return nMaxChild;
            }

            autoc nElementInNodeAvg = static_cast<float>(nElement) / static_cast<float>(nElementMax);
            autoc nDepthEstimated = std::min(nDepthMax, static_cast<depth_type>(std::ceil((log2f(nElementInNodeAvg) + 1.0) / static_cast<float>(nDimension))));
            if (nDepthEstimated * nDimension < 64)
                return static_cast<size_t>(rMult * (1 << nDepthEstimated * nDimension));

            return static_cast<size_t>(rMult * nElementInNodeAvg);
        }

        static inline depth_type EstimateMaxDepth(size_t nElement, max_element_type nElementMax) noexcept{
            if (nElement < nElementMax)
                return 2;

            autoc nLeaf = nElement / nElementMax;
            // nLeaf = (2^nDepth)^nDimension
            return std::clamp(static_cast<depth_type>(std::log2(nLeaf) / static_cast<double>(nDimension)), depth_type(2), depth_type(10));
        }

        static inline morton_node_id_type GetHash(depth_type depth, morton_node_id_type_cref key) noexcept{
            assert(key < (morton_node_id_type(1) << (depth * nDimension)));
            return (morton_node_id_type{ 1 } << (depth * nDimension)) | key;
        }

        static constexpr morton_node_id_type GetRootKey() noexcept{ 
            return morton_node_id_type{ 1 };
        }

        static constexpr bool IsValidKey(uint64_t key) noexcept { return key; }

        template<size_t N>
        static inline bool IsValidKey(bitset_arithmetic<N> const& key) noexcept { return !key.none(); }

        static depth_type GetDepth(morton_node_id_type key) noexcept{
            // Keep shifting off three bits at a time, increasing depth counter
            for (depth_type d = 0; IsValidKey(key); ++d, key >>= nDimension)
                if (key == 1) // If only sentinel bit remains, exit with node depth
                return d;

            assert(false); // Bad key
            return 0;
        }

        static inline morton_node_id_type RemoveSentinelBit(morton_node_id_type_cref key, std::optional<depth_type> const& onDepth = std::nullopt) noexcept{
            autoc nDepth = onDepth.has_value() ? *onDepth : GetDepth(key);
            return key - (morton_node_id_type{ 1 } << nDepth);
        }
    
    public: // Morton aid function

        static inline child_id_type getChildPartOfLocation(morton_node_id_type_cref key) noexcept{
            autoce maskLastBits1 = (morton_node_id_type{ 1 } << nDimension) - 1;
            return convertMortonIdToChildId(key & maskLastBits1);
        }

        static constexpr morton_grid_id_type part1By2(grid_id_type n) noexcept
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
            return std::is_same<morton_grid_id_type, bitset_arithmetic<nDimension>>::value ? morton_grid_id_type(n) : static_cast<morton_grid_id_type>(n);
        }

        // Separates low 16 bits of input by one bit
        static constexpr morton_grid_id_type part1By1(grid_id_type n) noexcept
        {
            // n = ----------------fedcba9876543210 : Bits initially
            // n = --------fedcba98--------76543210 : After (1)
            // n = ----fedc----ba98----7654----3210 : After (2)
            // n = --fe--dc--ba--98--76--54--32--10 : After (3)
            // n = -f-e-d-c-b-a-9-8-7-6-5-4-3-2-1-0 : After (4)
            n = (n ^ (n << 8)) & 0x00ff00ff; // (1)
            n = (n ^ (n << 4)) & 0x0f0f0f0f; // (2)
            n = (n ^ (n << 2)) & 0x33333333; // (3)
            n = (n ^ (n << 1)) & 0x55555555; // (4)
            return std::is_same<morton_grid_id_type, bitset_arithmetic<nDimension>>::value ? morton_grid_id_type(n) : static_cast<morton_grid_id_type>(n);
        }

    public: // Morton En- / Decoding functions

        static inline morton_grid_id_type MortonEncode(array<grid_id_type, nDimension> const& aidGrid) noexcept
        {
            if constexpr (nDimension == 1)
                return morton_grid_id_type(aidGrid[0]);
            else if constexpr (nDimension == 2)
                return (part1By1(aidGrid[1]) << 1) + part1By1(aidGrid[0]);
            else if constexpr (nDimension == 3)
                return (part1By2(aidGrid[2]) << 2) + (part1By2(aidGrid[1]) << 1) + part1By2(aidGrid[0]);
            else
            {
                auto msb = aidGrid[0];
                for (dim_type iDimension = 1; iDimension < nDimension; ++iDimension)
                msb |= aidGrid[iDimension];

                morton_grid_id_type id = 0;
                grid_id_type mask = 1;
                for (dim_type i = 0; msb; mask <<= 1, msb >>= 1, ++i)
                {
                    LOOPIVDEP
                    for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                    {
                        autoc shift = iDimension + i * nDimension;
                        id |= static_cast<morton_grid_id_type>(aidGrid[iDimension] & mask) << (shift - i);
                        
                    }
                }
                return id;
            }
        }

        static array<grid_id_type, nDimension> MortonDecode(morton_node_id_type_cref kNode, depth_type nDepthMax) noexcept
        {
            auto aidGrid = array<grid_id_type, nDimension>{};
            if constexpr (nDimension == 1)
                return { RemoveSentinelBit(kNode) };
            else
            {
                autoc nDepth = GetDepth(kNode);

                auto mask = morton_grid_id_type{ 1 };
                for (depth_type iDepth = nDepthMax - nDepth, shift = 0; iDepth < nDepthMax; ++iDepth){
                    for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension, ++shift){
                        aidGrid[iDimension] |= (kNode & mask) >> (shift - iDepth);
                        mask <<= 1;
                        
                    }
                    
                }
                    
            }
            return aidGrid;
        }

    public: // Getters
        
        inline auto const& GetNodes() const noexcept { return m_nodes; }
        inline auto & GetNodetoChange(morton_node_id_type_cref key) { return cont_at(m_nodes, key); }
        inline auto const& GetNode(morton_node_id_type_cref key) const noexcept { return cont_at(m_nodes, key); }
        inline auto const& GetParent(morton_node_id_type_cref key) const noexcept {return cont_at(m_nodes, key).m_parent;}
        inline auto const& GetBox() const noexcept { return m_box; }
        inline auto GetDepthMax() const noexcept { return m_nDepthMax; }
        inline auto GetResolutionMax() const noexcept { return m_nRasterResolutionMax; }

    public: // Main service functions

        // Initialises an empty tree
        constexpr void Init(box_type const& box, depth_type nDepthMax, max_element_type nElementMax = 11) noexcept
        {
            assert(this->m_nodes.empty()); // To build/setup/create the tree, use the Create() [recommended] or Init() function. If an already builded tree is wanted to be reset, use the Reset() function before init.
            assert(nDepthMax > 1);
            assert(nDepthMax <= m_nDepthMaxTheoretical);
            assert(nDepthMax < std::numeric_limits<uint8_t>::max());
            assert(nElementMax > 1);
            assert(CHAR_BIT * sizeof(grid_id_type) >= m_nDepthMax);

            this->m_box = box;
            this->m_rVolume = 1.0;
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                this->m_rVolume *= AD::point_comp_c(AD::box_max_c(this->m_box), iDimension) - AD::point_comp_c(AD::box_min_c(this->m_box), iDimension);

            this->m_nDepthMax = nDepthMax;
            this->m_nRasterResolutionMax = static_cast<grid_id_type>(pow_ce(2, nDepthMax));
            this->m_idSlotMax = this->m_nRasterResolutionMax - 1;
            this->m_nElementMax = nElementMax;

            auto& nodeRoot = this->m_nodes[GetRootKey()];
            nodeRoot.box = box;
            tie(this->m_aRasterizer, this->m_aBoxSize) = this->getGridRasterizer(AD::box_min_c(this->m_box), AD::box_max_c(this->m_box), this->m_nRasterResolutionMax);

            LOOPIVDEP
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                this->m_aMinPoint[iDimension] = static_cast<double>(AD::point_comp_c(AD::box_min_c(this->m_box), iDimension));
        }

        using fnProcedure = std::function<void(morton_node_id_type_cref, Node const&)>;
        using fnProcedureUnconditional = std::function<void(morton_node_id_type_cref, Node const&, bool)>;
        using fnSelector = std::function<bool(morton_node_id_type_cref, Node const&)>;
        using fnSelectorUnconditional = std::function<bool(morton_node_id_type_cref, Node const&)>;

        void VisitNodes(morton_node_id_type_cref kRoot, fnProcedure const& procedure, fnSelector const& selector) const noexcept
        {
            auto q = queue<morton_node_id_type>();
            
            for (q.push(kRoot); !q.empty(); q.pop()){
                
                autoc& key = q.front();
                autoc& node = cont_at(m_nodes, key);
                procedure(key, node);

                for (morton_node_id_type_cref kChild : node.GetChildren()){
                
                    if (selector(kChild, cont_at(m_nodes, kChild)))
                        q.push(kChild);
                
                }

            }
        }

        void PrintStructure(morton_node_id_type_cref kRoot = GetRootKey()) const noexcept{
            
            VisitNodes(kRoot, [&](morton_node_id_type_cref key, Node const& node){
                  
                // Node info
                std::cout << "Node ID = " << key << std::endl;
                std::cout << "Depth = " << int(GetDepth(key)) << std::endl;
                std::cout << "Parent ID = " << this->GetParent(key) << std::endl;
                std::cout << "Grid ID = " << "[ ";
                for(int i=0;i<3;++i) std::cout << MortonDecode(key,GetDepth(key))[i] << " ";
                std::cout << "]" << std::endl;


                // Box extent
                std::cout << " Box = [(";
                for(int i=0;i<3;++i) std::cout << node.box.Min[i] << ",";
                std::cout << "),(";                 
                for(int i=0;i<3;++i) std::cout << node.box.Max[i] << ",";
                std::cout << ")]" << std::endl;


                // Coarse Nbr
                if(key!=kRoot){
                    std::cout << "Coarse Nbrs : ";
                    auto coarse_nbrs = this->GetCoarseNeighbours(key);
                    for(auto cnb:coarse_nbrs) std::cout << cnb << " ";
                    std::cout << std::endl;
                }
                  

                // Colleagues
                std::cout <<"Colleagues : ";
                auto nbrs = this->GetColleagues(key);
                for(auto nb:nbrs) std::cout << nb << " ";
                std::cout<<std::endl;


                // Children
                std::cout << "Children : ";
                auto children = node.GetChildren();
                for(auto child : children) std::cout << child << " ";
                std::cout<<std::endl;


                // Points
                std::cout << "Points : ";
                if(node.IsAnyChildExist()){
                    auto points = this->CollectAllIdInBFS(key);  
                    for(auto point : points) std::cout << point << " ";
                }
                else{
                    for(int i=0;i<node.vid.size();i++){
                        std::cout<<node.vid.at(i)<<" ";
                    }
                }
                std::cout << std::endl;
                std::cout << std::endl << std::endl;

                
                }// end of functor
            ); 
            return;
        }

        std::vector<morton_node_id_type> GetLeafNodes(morton_node_id_type kRoot=GetRootKey()) const noexcept{
            
            std::vector<morton_node_id_type> leafnodes = {};
            VisitNodes(kRoot, [&](morton_node_id_type_cref key, Node const& node){ if(!node.IsAnyChildExist()){leafnodes.push_back(key);} });
        
            return leafnodes;
        }

        /**
         * @fn GetColleagues 
         * @param key morton id of node
         * @return vector of ids of colleagues of node with @param key
         * ! ONLY WORK FOR THE 3D CASE 
        */
        std::vector<morton_node_id_type_cref> GetColleagues(morton_node_id_type_cref key) const noexcept{
            
            auto aid = this->MortonDecode(key,GetDepth(key));
            std::vector<morton_node_id_type_cref> neighbours={};

            for(int x=-1;x<2;++x){
                for(int y=-1;y<2;++y){
                    for(int z=-1;z<2;++z){
                        auto nbr = this->MortonEncode(std::array{aid[0]+x, aid[1]+y, aid[2]+z})+std::pow(8,GetDepth(key));
                        if(m_nodes.contains(nbr)) neighbours.push_back(nbr); 
                    } 
                }
            }

            return neighbours;
        }

        /**
         * @fn GetPotentialColleagues
         * @param key morton id of node
         * @return vector of ids of potential colleagues (might not exists)
         * ! ONLY WORKDS FOR 3D CASE
        */
        std::vector<morton_node_id_type_cref> GetPotentialColleagues(morton_node_id_type_cref key) const noexcept{
            
            auto aid = this->MortonDecode(key,GetDepth(key));
            std::vector<morton_node_id_type_cref> neighbours={};
            
            for(int x=-1;x<2;++x){
                for(int y=-1;y<2;++y){
                    for(int z=-1;z<2;++z){
                        auto newaid = std::array{aid[0]+x, aid[1]+y, aid[2]+z};
                        if (not(newaid[0]<0 || newaid[0]>=std::pow(2,GetDepth(key)) || newaid[1]<0 || newaid[1]>=std::pow(2,GetDepth(key)) || newaid[2]<0 || newaid[2]>=std::pow(2,GetDepth(key)))){
                            auto nbr = this->MortonEncode(newaid)+std::pow(8,GetDepth(key));
                            neighbours.push_back(nbr);
                        }
                    } 
                }
            }

            return neighbours;
        }

        /**
         * @fn GetCoarseNeighbours
         * @param key morton id
         * @return vector of ids of coarse neighbours
        */
        std::vector<morton_node_id_type_cref> GetCoarseNeighbours(morton_node_id_type_cref key) const noexcept{
            
            auto node = this->GetNode(key);
            vector<morton_node_id_type_cref> coarse_neighbours{};
            auto parent_colleagues = this->GetColleagues(this->GetParent(key));

            for(auto parent_neighbour:parent_colleagues){
                
                //check if it is a leaf node and if it overlaps with the box
                auto coarse_node = this->GetNode(parent_neighbour);
                if(!coarse_node.IsAnyChildExist() && AD::box_relation(coarse_node.box, node.box)==AD::EBoxRelation::Adjecent){
                    coarse_neighbours.push_back(parent_neighbour);
                }
            }
            return coarse_neighbours;
        }
        
        /**
         * @fn GetNextAncestor
         * @param key morton id
         * @return morton id of next existing ancestor
        */
        morton_node_id_type_cref GetNextAncestor(morton_node_id_type_cref key) const noexcept{
            
            if(m_nodes.contains(key)) 
                return key;

            else if(key==0) 
                return 1;

            else
                return GetNextAncestor(key>>3);
        }
    
        /**
         * @fn CollectAllIdInBFS
         * @param kRoot key of node to start from (defaults to the root)
         * @return vector of point ids
        */
        vector<entity_id_type> CollectAllIdInBFS(morton_node_id_type_cref kRoot = GetRootKey()) const noexcept
        {
            auto ids = vector<entity_id_type>();
            ids.reserve(m_nodes.size() * std::max<size_t>(2, m_nElementMax / 2));

            VisitNodes(kRoot, [&ids](morton_node_id_type_cref, autoc& node)
            { 
                ids.insert(std::end(ids), std::begin(node.vid), std::end(node.vid));
            });
            return ids;
        }
        
        box_type CalculateExtent(morton_node_id_type_cref keyNode) const noexcept
        {
            auto boxNode = box_type();
            auto& ptMinBoxNode = AD::box_min(boxNode);
            auto& ptMaxBoxNode = AD::box_max(boxNode);
            autoc& ptMinBoxRoot = AD::box_min_c(m_box);
            autoc& ptMaxBoxRoot = AD::box_max_c(m_box);

            ptMinBoxNode = ptMinBoxRoot;

            auto aSize = array<geometry_type, nDimension>();
            LOOPIVDEP
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                aSize[iDimension] = AD::point_comp_c(ptMaxBoxRoot, iDimension) - AD::point_comp_c(ptMinBoxRoot, iDimension);

            autoc nDepth = GetDepth(keyNode);
            autoc nRasterResolution = pow_ce(2, nDepth);
            autoc rMax = 1.0 / static_cast<double>(nRasterResolution);

            autoce one = morton_grid_id_type{ 1 };
            auto keyShifted = keyNode;// RemoveSentinelBit(key, nDepth);
            for (depth_type iDepth = 0; iDepth < nDepth; ++iDepth)
            {
                autoc r = rMax * (1 << iDepth);

                LOOPIVDEP
                for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                {
                    autoc fApply = ((keyShifted >> iDimension) & one) > morton_grid_id_type{};
                    AD::point_comp(ptMinBoxNode, iDimension) += static_cast<geometry_type>((aSize[iDimension] * r)) * fApply;
                }
                keyShifted >>= nDimension;
            }

            LOOPIVDEP
            for (dim_type iDimension = 0; iDimension < nDimension; ++iDimension)
                AD::point_comp(ptMaxBoxNode, iDimension) = AD::point_comp_c(ptMinBoxNode, iDimension) + static_cast<geometry_type>(aSize[iDimension] * rMax);

            return boxNode;
        }

        void Reset() noexcept
        {
            m_nodes.clear();
            m_box = {};
            m_rVolume = 0.0;
            m_aRasterizer = {};
        }

        void Clear() noexcept
        {
            std::erase_if(m_nodes, [](autoc& p) { return p.first != GetRootKey(); });
            cont_at(m_nodes, GetRootKey()).vid.clear();
        }

        /**
         * @fn Move - Moves the whole tree by @param vMove
         * @param vMove the vector by which the tree is shifted
        */
        void Move(vector_type const& vMove) noexcept
        {
            //auto ep = execution_policy_type{}; // GCC 11.3
            std::for_each(/*ep,*/ std::begin(m_nodes), std::end(m_nodes), [&vMove](auto& pairKeyNode)
            {
                AD::move_box(pairKeyNode.second.box, vMove);
            });
            AD::move_box(this->m_box, vMove);
        }

        morton_node_id_type FindSmallestNodeKey(morton_node_id_type keySearch) const noexcept
        {
            for (; IsValidKey(keySearch); keySearch >>= nDimension)
                if (this->m_nodes.contains(keySearch))
                return keySearch;

            return morton_node_id_type{}; // Not found
        }

        morton_node_id_type Find(entity_id_type id) const noexcept
        {
            autoc it = find_if(this->m_nodes.begin(), this->m_nodes.end(), [id](autoc& keyAndNode)
            {
                return std::ranges::find(keyAndNode.second.vid, id) != end(keyAndNode.second.vid);
            });

            return it == this->m_nodes.end() ? 0 : it->first;
        }

    }; // class Orthotreebase



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