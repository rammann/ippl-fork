//
// unit test of morton_codes.h

#include "gtest/gtest.h"
#include "OrthoTree/MortonHelper.h"
#include <algorithm>
#include <cstdint>
#include <bitset>


TEST(MortonCodesTest, Encode3D) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  std::array<int, 3> coords = { 1, 2, 4 };
  int64_t code = morton.encode(coords, 3);
  int64_t expected = 0b1000100010011;
  EXPECT_EQ(code, expected);
}

TEST(MortonCodesTest, Decode3D) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t code = 0b1000100010011;
  std::array<int, 3> coords = morton.decode(code);
  std::array<int, 3> expected = {1, 2, 4};
  EXPECT_EQ(coords, expected);
}

TEST(MortonCodesTest, isDescendantTest) {

  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t root = 0;
  int64_t parent = 0b111;
  int64_t child1 = 0b01;
  int64_t child2 = 0b0011000;
  int64_t child3 = 0b1111111000;

  EXPECT_FALSE(morton.is_descendant(child1, parent));
  EXPECT_TRUE(morton.is_descendant(child2,parent));
  EXPECT_FALSE(morton.is_descendant(child3, parent));

  EXPECT_TRUE(morton.is_descendant(child1, root));
  EXPECT_TRUE(morton.is_descendant(child2, root));
  EXPECT_TRUE(morton.is_descendant(child3, root));
  EXPECT_TRUE(morton.is_descendant(parent, root));


}
TEST(MortonCodesTest, GetDeepestFirstChildTest) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t root = 0;
  int64_t child = morton.get_deepest_first_descendant(root);
  int64_t expected = 8;
  EXPECT_EQ(child, expected);
}


TEST(MortonCodesTest, GetDeepestLastChildTest) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t root = 0;
  int64_t child = morton.get_deepest_last_descendant(root);
  int64_t expected = morton.encode({255, 255, 255}, 8);
  EXPECT_EQ(child, expected);
}

TEST(MortonCodesTest, GetNearestCommonAncestorTest) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t code_a = morton.encode({124, 124, 124}, 6);
  int64_t code_b = morton.encode({0, 0, 0}, 8);
  int64_t ancestor = morton.get_nearest_common_ancestor(code_a, code_b);
  int64_t expected = morton.encode({0, 0, 0}, 1);
  EXPECT_EQ(ancestor, expected);
}

TEST(MortonCodesTest, GetChildrenTest) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t parent = morton.encode({126, 126, 126}, 7);
  vector_t<morton_code> children = morton.get_children(parent);
  vector_t<morton_code> expected;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      for (int k = 0; k < 2; k++) {
        expected.push_back(morton.encode({126 + i, 126 + j, 126 + k}, 8));
      }
    }
  }

  std::sort(expected.begin(), expected.end());

  EXPECT_EQ(children, expected);
}

TEST(MortonCodesTest, GetParentTest) {
  const size_t max_depth = 8;
  Morton<3> morton(max_depth);

  int64_t child = morton.encode({126, 126, 126}, 7);
  int64_t parent = morton.get_parent(child);
  int64_t expected = morton.encode({124, 124, 124}, 6);
  EXPECT_EQ(parent, expected);
}

