file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding unit tests found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

add_executable (morton_test morton_codes.cpp)
gtest_discover_tests (morton_test PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
target_link_libraries (
    morton_test
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (tree_test ortho_tree.cpp)
# gtest_discover_tests (tree_test PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
target_link_libraries (
    tree_test
    ippl
    pthread
    GTest::gtest_main
)

add_executable (parallel_tree_test parallel_ortho_tree.cpp)
target_link_libraries (
    parallel_tree_test
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (bounds_test bounding_box.cpp)
gtest_discover_tests (tree_test PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
target_link_libraries (
    bounds_test
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (linearise_test linearise.cpp)
gtest_discover_tests (linearise_test PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
target_link_libraries (linearise_test
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

add_executable (complete_region_test complete_region.cpp)
gtest_discover_tests (complete_region_test PROPERTIES TEST_DISCOVERY_TIMEOUT 600)
target_link_libraries (complete_region_test
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)




# vi: set et ts=4 sw=4 sts=4:

# Local Variables:
# mode: cmake
# cmake-tab-width: 4
# indent-tabs-mode: nil
# require-final-newline: nil
# End:
