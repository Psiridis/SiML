cmake_minimum_required(VERSION 3.28)
project(SiML_Library VERSION 1.0.0 LANGUAGES CXX)

set(SiML_LIB SiML)

option(SiML_ENABLE_TESTS "Enable building tests" ON)

set(SiML_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/temporary.cpp
    # Add more source files here
)

if(NOT SiML_SRC_FILES)
    message(FATAL_ERROR "Undefined SiML_SRC_FILES. Check source file list.")
endif()

add_library(${SiML_LIB} STATIC 
    ${SiML_SRC_FILES}
)

target_compile_features(${SiML_LIB} PUBLIC cxx_std_17)

# cache variables for installation destinations
include(GNUInstallDirs)

target_include_directories(${SiML_LIB}
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS ${SiML_LIB}
    EXPORT SiMLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SiMLTargets
    FILE SiMLTargets.cmake
    NAMESPACE ${SiML_LIB}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SiML_LIB}
)

# include the package config helpers and generate the config + version files
include(CMakePackageConfigHelpers)

configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SiML_LIB}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SiML_LIB}
)

if(SiML_ENABLE_TESTS)
  include(CTest)  # Enables testing infrastructure
  enable_testing()


  include(FetchContent)
  
  # Download GoogleTest at configure time
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  
  if (MSVC)
    # Prevent GoogleTest from overriding your compiler/linker options
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  endif()

  FetchContent_MakeAvailable(googletest)
  
  # Add your test executable
  add_executable(SiML_tests
      tests/test_temporary.cpp
      # Add more test source files here
  )
  
  target_link_libraries(SiML_tests
      PRIVATE
      ${SiML_LIB}
      gtest_main
  )
  
  target_include_directories(SiML_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
  
  # Register the test
  include(GoogleTest)
  gtest_discover_tests(SiML_tests)

endif()