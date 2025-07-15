# cache variables for installation destinations
include(GNUInstallDirs)

target_include_directories(${SiML_LIB}
    PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS ${SiML_LIB}
    EXPORT SiMLTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES include/printing.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT SiMLTargets
    FILE SiMLTargets.cmake
    NAMESPACE ${SiML_LIB}::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SiML_LIB}
)

# include the package config helpers and generate the file below
include(CMakePackageConfigHelpers)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SiML_LIB}
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/SiMLConfig.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${SiML_LIB}
)