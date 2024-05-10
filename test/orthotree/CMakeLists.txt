file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
message (STATUS "Adding index test found in ${_relPath}")

include_directories (
    ${CMAKE_SOURCE_DIR}/src
)

link_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Kokkos_DIR}/..
)

set (IPPL_LIBS ippl ${MPI_CXX_LIBRARIES})
set (COMPILE_FLAGS ${OPAL_CXX_FLAGS})

add_executable (TestOrthoTree TestOrthoTree.cpp)
add_executable (TestOrthoTreeSolver TestOrthoTreeSolver.cpp)
add_executable (TestGaussianTreeSolver TestGaussianTreeSolver.cpp)
add_executable(TestTreeConvergence TestTreeConvergence.cpp)
target_link_libraries (TestOrthoTree ${IPPL_LIBS})
target_link_libraries (TestOrthoTreeSolver ${IPPL_LIBS})
target_link_libraries (TestGaussianTreeSolver ${IPPL_LIBS})
target_link_libraries(TestTreeConvergence ${IPPL_LIBS})