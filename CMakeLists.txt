cmake_minimum_required(VERSION 3.3)

project(Simulator CXX)
set(CMAKE_CXX_STANDARD 17)

enable_testing()

add_subdirectory(LibSimulator)
add_subdirectory(Simulator)
add_subdirectory(TestSimulator)