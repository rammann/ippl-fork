//
// unit test of morton_codes.h

#include "gtest/gtest.h"

#include "OrthoTree/BoundingBox.h"

#include <algorithm>
#include <cstdint>
#include <bitset>

using namespace ippl;

// i will maybe move this to OrthoTree/utils.h or smth
template <size_t Dim>
bool compare_coordinates(const real_coordinate_template<Dim>& a, const real_coordinate_template<Dim>& b)
{
    for ( size_t i = 0; i < Dim; ++i ) {
        if ( a[i] != b[i] ) {
            return false;
        }
    }

    return true;
}

// i will maybe move this to OrthoTree/utils.h or OrthoTree/BoundingBox.h as a comparator
template <size_t Dim>
bool compare_boxes(const BoundingBox<Dim>& a, const BoundingBox<Dim>& b)
{
    return compare_coordinates<Dim>(a.get_min(), b.get_min())
        && compare_coordinates<Dim>(a.get_max(), b.get_max());
}

TEST(BoundingBoxTest, InitializeTest)
{
    // testing if a box is initialized to zero
    static constexpr size_t Dim = 3;
    BoundingBox<Dim> result;
    BoundingBox<Dim> expected({ 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.0 });

    EXPECT_EQ(compare_boxes(result, expected), true);
}

TEST(BoundingBoxTest, GetCenterTest)
{
    // testing if the center is calculated correctly
    static constexpr size_t Dim = 3;
    BoundingBox<Dim> box({ 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 });
    const auto result = box.get_center();
    real_coordinate_template<Dim> expected({ 0.5, 0.5, 0.5 });

    EXPECT_EQ(compare_coordinates<Dim>(result, expected), true);
}

TEST(BoundingBoxTest, ConvertIntegerGridToReal)
{
    // this will test the conversion from a grid coordinate to a real valued coordinate inside the root bounds, not implemented yet
    static constexpr size_t Dim = 3;

    BoundingBox<Dim> root_bounds({ 0.0, 0.0, 0.0 }, { 1.0, 1.0, 1.0 });

    //    EXPECT_EQ(compare_coordinates<Dim>(result, expected), true);
}
