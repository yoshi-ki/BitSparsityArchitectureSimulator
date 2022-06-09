add_executable(Simulator main.cpp)
add_executable(PerformanceTest performanceTest.cpp)
target_compile_options(PerformanceTest PUBLIC -O3 -Wall)

find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(PerformanceTest PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(PerformanceTest Python3::Python Python3::NumPy)

target_link_libraries(Simulator LibSimulator)
target_link_libraries(PerformanceTest LibSimulator)
