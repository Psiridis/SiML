include(CTest)
enable_testing()
include(FetchContent)

# GoogleTest
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
if (MSVC)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()
FetchContent_MakeAvailable(googletest)

# Test executable
add_executable(SiML_tests
    ${CMAKE_SOURCE_DIR}/tests/test_linear_regression_model.cpp
    ${CMAKE_SOURCE_DIR}/tests/test_MSE_loss_function.cpp
    ${CMAKE_SOURCE_DIR}/tests/test_Gradient_Descent_optimizer.cpp
)

target_link_libraries(SiML_tests PRIVATE ${SiML_LIB} gtest_main)

# This should not be necessary if ${SiML_LIB} already exports includes,
# but left here in case tests include Eigen directly
target_include_directories(SiML_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${eigen_SOURCE_DIR}
)

include(GoogleTest)
gtest_discover_tests(SiML_tests)