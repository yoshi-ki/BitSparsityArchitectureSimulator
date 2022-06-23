add_library(LibSimulator PE.cpp PEArray.cpp Utils.cpp)

target_include_directories(LibSimulator PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(LibSimulator PUBLIC -O3 -Wall)
