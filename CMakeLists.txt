cmake_minimum_required(VERSION 3.28)
project(SiML_Library VERSION 1.0.0 LANGUAGES CXX)

set(SiML_LIB SiML)

option(SiML_ENABLE_TESTS "Enable building tests" ON)

# Dependencies (Eigen, etc.)
include(cmake/Dependencies.cmake)

# Build the main library
add_subdirectory(src)

# Install rules + CMake package config
include(cmake/InstallConfig.cmake)

# Tests
if(SiML_ENABLE_TESTS)
    include(cmake/Tests.cmake)
endif()