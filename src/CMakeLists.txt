set(SiML_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/loss_function.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/optimizer.cpp
)

if(NOT SiML_SRC_FILES)
    message(FATAL_ERROR "No source files found for SiML library")
endif()

add_library(${SiML_LIB} STATIC ${SiML_SRC_FILES})

target_compile_features(${SiML_LIB} PUBLIC cxx_std_20)

target_include_directories(${SiML_LIB}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${eigen_SOURCE_DIR}
)