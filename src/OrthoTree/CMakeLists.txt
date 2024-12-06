set (_SRCS
helpers/AidList.cpp
    )

set (_HDRS
    OrthoTreeParticle.h
    Types.h
    MortonHelper.hpp
    OrthoTree.h
    OrthoTree.hpp
    MortonHelper.h
    BoundingBox.h
    BoundingBox.hpp
helpers/AidList.h
helpers/AidList.hpp
    )

include_directories (
    ${CMAKE_CURRENT_SOURCE_DIR}
    )

ADD_IPPL_SOURCES (${_SRCS})
ADD_IPPL_HEADERS (${_HDRS})

install (FILES ${_HDRS} DESTINATION include/OrthoTree)
