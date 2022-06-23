find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(Simulator main.cpp)
add_executable(PerformanceTest performanceTest.cpp)
target_compile_options(PerformanceTest PUBLIC -O3 -Wall -fopenmp -lomp)

find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(PerformanceTest PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(PerformanceTest Python3::Python Python3::NumPy)

target_link_libraries(Simulator LibSimulator)
target_link_libraries(PerformanceTest LibSimulator)
