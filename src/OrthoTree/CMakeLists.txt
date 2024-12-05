set (_SRCS

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
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/OrthoTree)
add_compile_definitions(IPPL_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
