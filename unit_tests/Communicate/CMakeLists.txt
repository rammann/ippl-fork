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

add_executable (BufferHandler BufferHandler.cpp)
gtest_discover_tests(BufferHandler PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

add_executable (LoggingBufferHandler LoggingBufferHandler.cpp)
gtest_discover_tests(LoggingBufferHandler PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

add_executable (LogEntry LogEntry.cpp)
gtest_discover_tests(LogEntry PROPERTIES TEST_DISCOVERY_TIMEOUT 600)

target_link_libraries (
    BufferHandler 
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

target_link_libraries (
    LoggingBufferHandler 
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)

target_link_libraries (
    LogEntry 
    ippl
    pthread
    GTest::gtest_main
    ${MPI_CXX_LIBRARIES}
)
