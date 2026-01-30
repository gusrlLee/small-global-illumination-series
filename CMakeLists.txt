cmake_minimum_required(VERSION 3.30 FATAL_ERROR)
project(small-global-illumination-series LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES native)

message(STATUS "---------------------------------------------------")
message(STATUS ">> Current CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS ">> CUDA Compiler Version: ${CMAKE_CUDA_COMPILER_VERSION}")
message(STATUS "---------------------------------------------------")


add_subdirectory(ext)

add_subdirectory(small_pt)
add_subdirectory(small_nee)
add_subdirectory(small_mis)
add_subdirectory(tinypt-cu)
add_subdirectory(tinypt-lbvh)
add_subdirectory(tinypt-optix)
add_subdirectory(tinypt-wavefront)