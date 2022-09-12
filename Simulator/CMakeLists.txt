find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(Torch REQUIRED)
add_executable(easytorch easytorch.cpp)
target_link_libraries(easytorch "${TORCH_LIBRARIES}")
# set_property(TARGET easytorch PROPERTY CXX_STANDARD 14)

add_executable(easytorchbfloat easytorchbfloat.cpp)
target_link_libraries(easytorchbfloat "${TORCH_LIBRARIES}")

add_executable(tryApprox tryApprox.cpp)
target_link_libraries(tryApprox "${TORCH_LIBRARIES}")

add_executable(Simulator main.cpp)
add_executable(PerformanceTest performanceTest.cpp)
target_compile_options(PerformanceTest PUBLIC -O3 -Wall -fopenmp -lomp -mtune=native -march=native)

find_package(Python3 COMPONENTS Development NumPy)
target_include_directories(PerformanceTest PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(PerformanceTest Python3::Python Python3::NumPy)

target_link_libraries(tryApprox LibSimulator)
target_link_libraries(easytorchbfloat LibSimulator)
target_link_libraries(easytorch LibSimulator)
target_link_libraries(Simulator LibSimulator)
target_link_libraries(PerformanceTest LibSimulator)