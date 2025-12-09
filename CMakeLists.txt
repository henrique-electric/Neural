cmake_minimum_required(VERSION 4.1.0)
set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_CXX_STANDARD 23)

project("NN" CXX)

find_package(eigen3 CONFIG REQUIRED)

include_directories(./include)

add_executable(Main src/main.cpp src/nn.cpp)
target_link_libraries(Main PRIVATE Eigen3::Eigen)