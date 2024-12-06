set (_SRCS
helpers/AidList.cpp
    )

set (_HDRS
    Types.h
    OrthoTreeParticle.h
    OrthoTree.h
    OrthoTree.hpp
    helpers/MortonHelper.h
    helpers/MortonHelper.hpp
    helpers/BoundingBox.h
    helpers/BoundingBox.hpp
    helpers/AidList.h
    helpers/AidList.hpp
)

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR} # Add this for generated files
)

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

# Install headers
install (FILES ${_HDRS} DESTINATION include/OrthoTree)

# Define IPPL_SOURCE_DIR for use in code
add_compile_definitions(IPPL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

# Configure IPPLConfig.h
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/helpers/Config.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/helpers/Config.h)
