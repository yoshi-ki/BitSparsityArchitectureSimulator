add_library(LibSimulator PE.cpp PEArray.cpp Utils.cpp)

# find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

target_include_directories(LibSimulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(LibSimulator PUBLIC -O3 -Wall -fopenmp -lomp)
