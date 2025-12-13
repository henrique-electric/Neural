cmake_minimum_required(VERSION 4.1.0)
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_STANDARD 23)

project("NN" CXX)

find_package(eigen3 CONFIG REQUIRED)

include_directories(./include vcpkg_installed/arm64-osx/include/)

add_library(NN STATIC src/nn.cpp src/functions.cpp src/inits.cpp src/loss.cpp)

add_executable(Main src/main.cpp)
target_link_libraries(Main PRIVATE Eigen3::Eigen NN)
