cmake_minimum_required(VERSION 4.1.0)
set(CMAKE_CXX_STANDARD 23)

project("NN" CXX)

if (LINUX)
    set(VCPKG_PACKAGE_INCLUDE vcpkg_installed/x64-linux/)
elseif(APPLE)
    set(VCPKG_PACKAGE_INCLUDE vcpkg_installed/arm64-osx/)
endif()

find_package(Eigen3 CONFIG REQUIRED)

include_directories(./include ${VCPKG_PACKAGE_INCLUDE}/include/)

add_library(NN STATIC src/nn.cpp src/functions.cpp src/inits.cpp src/loss.cpp)

add_executable(Main src/main.cpp)
target_link_libraries(Main PRIVATE Eigen3::Eigen NN)

add_custom_target(run COMMAND $<TARGET_FILE:Main> DEPENDS Main WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
