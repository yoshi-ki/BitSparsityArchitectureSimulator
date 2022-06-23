cmake_minimum_required(VERSION 3.3)

# find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

project(Simulator CXX)
set(CMAKE_CXX_STANDARD 17)

enable_testing()

add_subdirectory(LibSimulator)
add_subdirectory(Simulator)
add_subdirectory(TestSimulator)
