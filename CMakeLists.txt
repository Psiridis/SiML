cmake_minimum_required(VERSION 3.28)
project(SiML_Library LANGUAGES CXX)

set(SiML_LIB SiML)

include(cmake_files/objects.in.cmake)

add_library(${SiML_LIB}
    ${SiML_SRC_FILES}
)

set_target_properties(${SiML_LIB}
    PROPERTIES
    CMAKE_CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED ON
    CMAKE_CXX_EXTENSIONS OFF
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include(cmake_files/installation_rules.cmake)