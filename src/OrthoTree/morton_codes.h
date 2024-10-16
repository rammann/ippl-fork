#ifndef MORTON_ENCODER_H  
#define MORTON_ENCODER_H

#include <array>
#include <vector>
#include <cstdint>

using morton_code = uint16_t;
using grid_t = int;

template <typename T>
using vector_t = std::vector<T>;

template <size_t Dim>
using grid_coordinates_template = std::array<grid_t, Dim>;

template <size_t Dim>
using real_coordinates_template = std::array<double, Dim>;

template <size_t Dim>
struct Morton {
    using grid_coordinates = grid_coordinates_template<Dim>;
    using real_coordinates = real_coordinates_template<Dim>;

public:
    static Morton& getInstance(size_t max_depth);

    // deleted to enforce singleton
    Morton(const Morton&) = delete;
    Morton& operator=(const Morton&) = delete;
    Morton(Morton&&) = delete;
    Morton& operator=(Morton&&) = delete;

    inline morton_code encode(const real_coordinates& coordinate, const real_coordinates& rasterizer, const size_t depth);
    inline morton_code encode(const grid_coordinates& coordinate, const size_t depth);

    inline grid_coordinates decode(morton_code code) const;

    inline size_t get_depth(morton_code code) const;

    inline morton_code get_parent(morton_code code) const;

    vector_t<morton_code> get_children(morton_code code) const;
    vector_t<morton_code> get_siblings(morton_code code) const;

    // dk whats going on here
    void get_descendant(morton_code code) const;

    morton_code get_first_child(morton_code code) const;
    morton_code get_last_child(morton_code code) const;

    morton_code get_first_descendant(morton_code code, const size_t level) const;
    morton_code get_last_descendant(morton_code code, const size_t level) const;

    morton_code get_deepest_first_descendant(morton_code code) const;
    morton_code get_deepest_last_descendant(morton_code code) const;

    // testing 
    morton_code get_ancestor_at_relative_level(morton_code code, size_t level) const
    {
        const morton_code depth = get_depth(code);
        assert(level <= depth && "cant go higher than root node!");

        return ((code >> (depth_mask_shift + (Dim * depth))) << (depth_mask_shift + (Dim * depth))) | (depth - 1);
    }

    morton_code get_nearest_common_ancestor(morton_code code_a, morton_code code_b) const;

    // what is meant by A(n) in the paper? get a list of all ancestors, or boolean value...?
    // TODO morton_code get_ancestor(morton_code code) const; 
    // TODO bool is_ancestor(morton_code child, morton_code parent) const; 

    // TODO vector_t<morton_code> get_list_potential_neighbors(morton_code code, const size_t level) const; 
    // TODO vector_t<morton_code> get_list_potential_neighbors_sharing_corner(morton_code code, const size_t level) const; 
    // TODO vector_t<morton_code> get_neighbors(morton_code code) const; 
    // TODO vector_t<morton_code> get_insulation_layer(morton_code code) const; 

private:
    const size_t max_depth;
    const size_t depth_mask_shift;
    const size_t depth_mask;
    const size_t n_children;

    // private to enforce singleton
    Morton(size_t max_depth)
        : max_depth(max_depth),
        depth_mask_shift(std::floor(std::log2(max_depth)) + 1),
        depth_mask((1 << depth_mask_shift) - 1),
        n_children((1 << (Dim)))
    { }

    // gets the step size between siblings
    inline morton_code get_step_size(morton_code code) const;

    // spreads one coordinate axis onto a morton code
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
    for ( size_t i = 0; i < Dim; ++i ) {
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
        for ( size_t bit = 0; bit < (sizeof(morton_code) * 8) / 3; ++bit ) {
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

    const morton_code cur_shift = (Dim * (max_depth - depth + 1)) + depth_mask_shift;

    // this removes the depth bits
    const morton_code parent_code = code >> cur_shift;

    return (parent_code << cur_shift) | parent_depth_bits;
}

template <size_t Dim>
vector_t<morton_code> Morton<Dim>::get_children(morton_code code) const
{
    const morton_code first_child = get_first_child(code);
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

template <size_t Dim>
morton_code Morton<Dim>::get_first_descendant(morton_code code, const size_t level) const
{
    return code + level;
}

template <size_t Dim>
morton_code Morton<Dim>::get_last_descendant(morton_code code, const size_t level) const
{
    size_t current_depth = get_depth(code);
    size_t total_levels = level - current_depth;

    size_t num_descendants = 0;
    for ( size_t i = 1; i <= total_levels; ++i ) {
        num_descendants += static_cast<size_t>(pow(n_children, i));
    }

    return code + num_descendants;
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

template <size_t Dim>
morton_code Morton<Dim>::get_nearest_common_ancestor(morton_code code_a, morton_code code_b) const
{
    morton_code ancestor_a = code_a;
    morton_code ancestor_b = code_b;

    size_t depth_a = get_depth(ancestor_a);
    size_t depth_b = get_depth(ancestor_b);

    // bring both nodes to the same depth
    while ( depth_a > depth_b ) {
        ancestor_a = get_parent(ancestor_a);
        depth_a--;
    }

    while ( depth_b > depth_a ) {
        ancestor_b = get_parent(ancestor_b);
        depth_b--;
    }

    // climp up until common ancestor is found
    while ( ancestor_a != ancestor_b ) {
        assert(ancestor_a != 0 && ancestor_b != 0 && "cant ascend more than root node");
        ancestor_a = get_parent(ancestor_a);
        ancestor_b = get_parent(ancestor_b);
    }

    return ancestor_a;
}

template <size_t Dim>
inline morton_code Morton<Dim>::get_step_size(morton_code code) const
{
    return 1 << (Dim * (max_depth - get_depth(code) + 1));
}

template <size_t Dim>
inline morton_code Morton<Dim>::spread_coords(grid_t coord) const
{
    morton_code res = 0;
    for ( size_t i = 0; i < max_depth; ++i ) {
        const auto current_bit = (coord >> i) & 1ULL;
        const auto shift = (i * Dim);
        res |= (current_bit << shift);
    }

    return res;
}

#endif // MORTON_ENCODER_H