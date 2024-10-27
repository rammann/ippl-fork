#ifndef MORTON_ENCODER_H  
#define MORTON_ENCODER_H

#include <array>
#include <vector>
#include <cstdint>

using morton_code = uint16_t;
using grid_t = int;

// adjust for ippl/kokkos
template <typename T>
using vector_t = std::vector<T>;

// adjust for ippl/kokkos
template <size_t Dim>
using grid_coordinates_template = std::array<grid_t, Dim>;

// adjust for ippl/kokkos
template <size_t Dim>
using real_coordinates_template = std::array<double, Dim>;

/**
 * @brief This class manages morton codes for the octree.
 * The functions are implemented based on page 4 of https://padas.oden.utexas.edu/static/papers/OctreeBalance21.pdf
 *
 * The class is a singleton as of now, which is probably something we will have to change.
 * I would suggest that we keep the class design, this way we dont have to add parameters to each function call.
 * Maybe we can also represent a morton code as a class, this would make stuff like code_a.is_parent(potential_parent_code) cleaner.
 * Changing the framework of the functions is easy, so its not important.
 *
 * @tparam Dim how many dimensions our space has.
 */
template <size_t Dim>
struct Morton {
    // change this to work with ippl later
    using grid_coordinates = grid_coordinates_template<Dim>;
    // change this to work with ippl later
    using real_coordinates = real_coordinates_template<Dim>;

public:
    /**
     * @brief Singleton pattern instance getter -> will deliver the same instance to all callers (maybe dumb for hpc threads n stuff)
     *
     * @param max_depth
     * @return Morton&
     
    static Morton& getInstance(size_t max_depth);

    // deleted to enforce singleton
    Morton(const Morton&) = delete;
    Morton& operator=(const Morton&) = delete;
    Morton(Morton&&) = delete;
    Morton& operator=(Morton&&) = delete;

    /**
     * @brief Encodes the given coordinate based on the rasterizer. As of now, the rasterizer is used to transform the (real valued) coordinates to a grid
     * based coordinate system, i dont know if this is a good idea, but it works for spaces in the range [0,1]^Dim
     *
     * @param coordinate real values coordinate
     * @param rasterizer real values of one unit length -> should maybe be a boundingbox starting at min, min?
     * @param depth the expected depth of the code, used to scale the rasterizer
     *
     * @warning no checks are done (for anything lol) if the coordinate is encodable for the given grid!
     *
     * @return morton_code
     */
    inline morton_code encode(const real_coordinates& coordinate, const real_coordinates& rasterizer, const size_t depth);

    /**
     * @brief Encodes the given grid based coordinate to a morton code.
     *
     * @param coordinate grid based coordiante
     * @param depth the depth at which we want to encode the coordinate
     *
     * @warning no checks are done if the coordinate is encodable for the given grid!
     *
     * @return morton_code
     */
    inline morton_code encode(const grid_coordinates& coordinate, const size_t depth);

    /**
     * @brief Decodes the given morton code into an integer based coordiante vector
     *
     * @param code a valid morton code (also works with invalid codes lol)
     * @return grid_coordinates
     */
    inline grid_coordinates decode(morton_code code) const;

    /**
     * @brief Returns the encoded depth of a code
     *
     * @param code
     * @return size_t
     */
    inline size_t get_depth(morton_code code) const;

    /**
     * @brief Returns the code of this parent
     *
     * @param code
     * @return morton_code
     */
    inline morton_code get_parent(morton_code code) const;

    /**
     * @brief Returns a vector filled with the (2^Dim) children of a node, in ascending order
     *
     * @param code
     * @return vector_t<morton_code>
     */
    vector_t<morton_code> get_children(morton_code code) const;

    /**
     * @brief Returns the siblings of the given code in ascending order.
     * The function works by getting the codes parent and then generating its kids, meaning the siblings contain the given code itself
     *
     * @param code
     * @return vector_t<morton_code>
     */
    vector_t<morton_code> get_siblings(morton_code code) const;

    /**
     * @brief Returns the first child of a code
     *
     * @param code
     * @return morton_code
     */
    morton_code get_first_child(morton_code code) const;

    /**
     * @brief Returns the last child of a code
     *
     * @param code
     * @return morton_code
     */
    morton_code get_last_child(morton_code code) const;

    /**
     * @brief Returns the lth first descendand, the walk is done using only first descendants.
     *
     * @param code
     * @param level
     * @return morton_code
     */
    morton_code get_first_descendant(morton_code code, const size_t level) const;

    /**
     * @brief Returns the lth last descendant, the walk is done using only last descendants.
     *
     * @param code
     * @param level
     * @return morton_code
     */
    morton_code get_last_descendant(morton_code code, const size_t level) const;

    /**
     * @brief Returns get_first_descendant(code, max_depth - get_depth(code));
     *
     * @param code
     * @return morton_code
     */
    morton_code get_deepest_first_descendant(morton_code code) const;

    /**
     * @brief Returns get_first_descendant(code, max_depth - get_depth(code));
     *
     * @param code
     * @return morton_code
     */
    morton_code get_deepest_last_descendant(morton_code code) const;

    // testing 
    morton_code get_ancestor_at_relative_level(morton_code code, size_t level) const
    {
        const morton_code depth = get_depth(code);
        assert(level <= depth && "cant go higher than root node!");

        return ((code >> (depth_mask_shift + (Dim * depth))) << (depth_mask_shift + (Dim * depth))) | (depth - 1);
    }

    /**
     * @brief Returns the nearest common ancestor of the two given codes, is implemented very naively and can for sure be done smarter.
     *
     * @param code_a
     * @param code_b
     * @return morton_code
     */
    morton_code get_nearest_common_ancestor(morton_code code_a, morton_code code_b) const;

    // what is meant by A(n) in the paper? get a list of all ancestors, or boolean value...?
    // TODO 
    morton_code get_ancestor(morton_code code) const;
    /**
     * @brief Checks whether the given parent is an ancestor of the given child
     *
     * @param child the child code
     * @param parent the parent code
     * @return true if parent is ancestor of child
     */
    bool is_ancestor(morton_code child, morton_code parent) const;

    // what do they want here?
    // TODO 
    void get_descendant(morton_code code) const;
    /**
     * @brief Checks whether the given child is a descendant of the given parents
     *
     * @param child the child code
     * @param parent the parent code
     * @return true if child is descendant of parent
     */
    bool is_descendant(morton_code child, morton_code parent) const;

    // TODO 
    vector_t<morton_code> get_list_potential_neighbors(morton_code code, const size_t level) const;
    // TODO shared corner is corner shared with parent node, these eight neighbors don't have the same parent
    vector_t<morton_code> get_list_potential_neighbors_sharing_corner(morton_code code, const size_t level) const;
    // TODO 
    vector_t<morton_code> get_neighbors(morton_code code) const;

    /**
     * @brief Get the insulation layer object
     *
     * @param code
     * @return vector_t<morton_code>
     */
    //TODO
    vector_t<morton_code> get_insulation_layer(morton_code code) const;

private:
    const size_t max_depth;
    const size_t depth_mask_shift;
    const size_t depth_mask;
    const size_t n_children;

    Morton(size_t max_depth)
        : max_depth(max_depth),
        depth_mask_shift(std::floor(std::log2(max_depth)) + 1),
        depth_mask((1 << depth_mask_shift) - 1),
        n_children((1 << (Dim)))
    { }

    /**
     * @brief Returns the step size with siblings at a given level
     *
     * @param code
     * @return morton_code
     */
    inline morton_code get_step_size(morton_code code) const;

    /**
     * @brief Spreads the coordinates of a single axis onto a morton code. This means:
     * For dim = 2
     * coord = 1111, then spread_cord(coord) -> 01010101
     * This will be used to construct the full code in the encode function.
     *
     * @warning NOTE there is no shifting happening, all bits are spread equally, they are only shifted in encode.
     *
     * @param coord
     * @return morton_code
     */
    inline morton_code spread_coords(grid_t coord) const;
};

template <size_t Dim>
Morton<Dim>& Morton<Dim>::getInstance(size_t max_depth)
{
    static Morton<Dim> morton(max_depth);
    return morton;
}

template <size_t Dim>
morton_code Morton<Dim>::encode(const real_coordinates& coordinate, const real_coordinates& rasterizer, const size_t depth)
{
    grid_coordinates grid = { {} };

    // also: we use two for loops because encode is called instead of encoding directly.
    for ( size_t i = 0; i < Dim; ++i ) {
        // this will probably break with negative values or if 0 is not min of bounding box
        const double normalised_coords = (coordinate[i] / rasterizer[i]);
        grid[i] = static_cast<morton_code>(normalised_coords);
    }

    return encode(grid, depth);
}

template <size_t Dim>
morton_code Morton<Dim>::encode(const grid_coordinates& coordinate, const size_t depth)
{
    morton_code code = 0;
    for ( size_t i = 0; i < Dim; ++i ) {
        const morton_code spread_bits = spread_coords(coordinate[i]);
        code |= (spread_bits << i);
    }

    code = (code << depth_mask_shift) | depth;
    return code;
}

template <size_t Dim>
Morton<Dim>::grid_coordinates Morton<Dim>::decode(morton_code code) const
{
    code = code >> depth_mask_shift; // remove depth information

    grid_coordinates grid_pos { {} };
    for ( size_t i = 0; i < Dim; ++i ) {
        // limit amount of traversed bits
        for ( size_t bit = 0; bit < (sizeof(morton_code) * 8) / Dim; ++bit ) {
            const morton_code cur_bit = (code >> ((bit * Dim) + i)) & 1ULL;
            grid_pos[i] |= (cur_bit << bit);
        }
    }

    return grid_pos;
}

template <size_t Dim>
inline size_t Morton<Dim>::get_depth(morton_code code) const
{
    return code & depth_mask;
}

template <size_t Dim>
inline morton_code Morton<Dim>::get_parent(morton_code code) const
{
    const morton_code depth = get_depth(code);
    const morton_code parent_depth_bits = depth - 1;

    // the first part removes irellevant bits (basically only keeping bits that ALL descendants of a code share with its ancestor)
    // the last part removes the depth bits
    const morton_code cur_shift = (Dim * (max_depth - depth + 1)) + depth_mask_shift;

    // remove all irrelevant bits
    const morton_code parent_code = code >> cur_shift;

    // shift back, resulting in zeros, add parent depth information back in
    return (parent_code << cur_shift) | parent_depth_bits;
}

template <size_t Dim>
vector_t<morton_code> Morton<Dim>::get_children(morton_code code) const
{
    const morton_code first_child = get_first_child(code);

    // each level has a distinctive step size between siblings, this can maybe be improved upon
    const morton_code step = get_step_size(first_child);

    vector_t<morton_code> vec;
    vec.reserve(n_children);

    for ( size_t i = 0; i < n_children; ++i ) {
        vec.push_back(first_child + (i * step));
    }

    return vec;
}

template <size_t Dim>
vector_t<morton_code> Morton<Dim>::get_siblings(morton_code code) const
{
    return get_children(get_parent(code));
}

template <size_t Dim>
bool Morton<Dim>::is_descendant(morton_code child, morton_code parent) const  {

    // descendants are always larger than their parents
    if (child < parent) return false;

    const morton_code step = get_step_size(parent);
    const morton_code next_neighbour = parent + step;

    // if the child is a descendant the parent, is must be smaller than 
    // the parent's next bigger neighbour at the level of the parent
    return child < (next_neighbour);
}

template <size_t Dim>
bool Morton<Dim>::is_ancestor(morton_code child, morton_code parent) const
{
    return is_descendant(child, parent);
}


template <size_t Dim>
morton_code Morton<Dim>::get_first_child(morton_code code) const
{
    return code + 1;
}

template <size_t Dim>
morton_code Morton<Dim>::get_last_child(morton_code code) const
{
    const morton_code first_child = get_first_child(code);
    const morton_code step = get_step_size(first_child);
    return first_child + (n_children - 1) * step;
}

// implementation corrected for absolute level
template <size_t Dim>
morton_code Morton<Dim>::get_first_descendant(morton_code code, const size_t level) const
{
    return (code & (~depth_mask)) + level;
}

// implementation corrected for absolute level
template <size_t Dim>
morton_code Morton<Dim>::get_last_descendant(morton_code code, const size_t level) const
{
    const morton_code current_depth = get_depth(code);
    assert(level >= current_depth && 
        "can't get descendants at a coarser level than the current node!");
    const morton_code first_descendant = get_first_descendant(code, level); 
    const morton_code step = get_step_size(first_descendant);

    // the number of descendants at a given relative level are given by 
    // 2^(Dim * (level difference)) as each level multiplies a factor 2^Dim
    const morton_code num_descendants = (1 << (Dim * (level - current_depth)));

    // the last descendant is num_descendants - 1 morton code steps
    // away from the first descendant
    return first_descendant + step*(num_descendants - 1);
}

template <size_t Dim>
morton_code Morton<Dim>::get_deepest_first_descendant(morton_code code) const
{
    return get_first_descendant(code, max_depth);
}

template <size_t Dim>
morton_code Morton<Dim>::get_deepest_last_descendant(morton_code code) const
{
    return get_last_descendant(code, max_depth);
}


// suggestion for alternative algorithm:
// only go up the levels on the larger node and check whether the other is descendant.
template <size_t Dim>
morton_code Morton<Dim>::get_nearest_common_ancestor(morton_code code_a, morton_code code_b) const
{

    size_t depth_a = get_depth(code_a);
    size_t depth_b = get_depth(code_b);

    // swap nodes such that b is the coarser nodes
    if ( depth_a < depth_b ) {
        std::swap(code_a, code_b);
        std::swap(depth_a, depth_b);
    }


    morton_code ancestor_b = code_b;
    // climp up until common ancestor is found
    while ( !is_descendant(code_a, ancestor_b) ) {
        ancestor_b = get_parent(ancestor_b);
    }

    return ancestor_b;
}

template <size_t Dim>
inline morton_code Morton<Dim>::get_step_size(morton_code code) const
{
    // could it be that this can be simplified the following way:
    // the min step size is equal to floor(log2(max_depth)) + 1 == sizeof(depth encoding)
    // each level above min depth increases step size by Dim bits
    // so simplified: 1 << (depth_mask_shift + Dim * (max_depth - get_depth(code)))
    return 1 << (Dim * (max_depth - get_depth(code) + 1));
}

template <size_t Dim>
inline morton_code Morton<Dim>::spread_coords(grid_t coord) const
{
    morton_code res = 0;
    for ( size_t i = 0; i < max_depth; ++i ) {
        // should be right, idk if this is possible without a loop, to guarantee unroll: replace max_depth with sizeof(morton_code)
        const auto current_bit = (coord >> i) & 1ULL;
        const auto shift = (i * Dim);
        res |= (current_bit << shift);
    }

    return res;
}

#endif // MORTON_ENCODER_H