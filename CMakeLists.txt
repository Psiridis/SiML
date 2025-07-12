cmake_minimum_required(VERSION 3.28)
project(SiML_Library LANGUAGES CXX)

add_library(SiML
    "src/printing.cpp"
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set_target_properties(SiML
    PROPERTIES
    CMAKE_CXX_STANDARD 17
    CMAKE_CXX_STANDARD_REQUIRED ON
    CMAKE_CXX_EXTENSIONS OFF
)

# cache variables for installation destinations
include(GNUInstallDirs)

target_include_directories(SiML
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS SiML
    EXPORT SiMLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES include/printing.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SiMLTargets
    FILE SiMLTargets.cmake
    NAMESPACE SiML::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SiML
)

# include the package config helpers and generate the file below
include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SiML
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SiML
)