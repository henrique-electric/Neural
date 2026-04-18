cmake_minimum_required(VERSION 4.1.0)
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_STANDARD 23)

project("NN" CXX)

find_package(Eigen3 CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)

include_directories(./include vcpkg_installed/arm64-osx/include/ vcpkg_installed/x64-windows/include/ vcpkg_installed/x64-linux/include)

add_library(NN STATIC src/nn.cpp src/activations.cpp src/inits.cpp src/loss.cpp)

add_executable(Main src/main.cpp)
target_link_libraries(Main PRIVATE Eigen3::Eigen NN)

enable_testing()
include(GoogleTest)

add_executable(test_activation_functions tests/test_activation_functions.cpp)
target_link_libraries(test_activation_functions PRIVATE GTest::gtest_main Eigen3::Eigen NN)
gtest_discover_tests(test_activation_functions)

add_executable(test_loss_functions tests/test_loss_functions.cpp)
target_link_libraries(test_loss_functions PRIVATE GTest::gtest_main Eigen3::Eigen NN)
gtest_discover_tests(test_loss_functions)

add_executable(test_weight_initialization tests/test_weight_initialization.cpp)
target_link_libraries(test_weight_initialization PRIVATE GTest::gtest_main Eigen3::Eigen NN)
gtest_discover_tests(test_weight_initialization)

add_executable(test_neural_network tests/test_neural_network.cpp)
target_link_libraries(test_neural_network PRIVATE GTest::gtest_main Eigen3::Eigen NN)
gtest_discover_tests(test_neural_network)

# One-command target: build all tests and execute the full suite.
add_custom_target(run_all_tests
	COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
	DEPENDS
		test_activation_functions
		test_loss_functions
		test_weight_initialization
		test_neural_network
	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
	USES_TERMINAL
)
