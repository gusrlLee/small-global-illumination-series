cmake_minimum_required(VERSION 3.30)
set(SAMPLE_NAME "tinypt-cu")

file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.h" 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh"
)

add_executable(${SAMPLE_NAME} ${SOURCES})
target_link_libraries(${SAMPLE_NAME} tinyobjloader)
target_include_directories(${SAMPLE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext)