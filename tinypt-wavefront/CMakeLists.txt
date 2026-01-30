cmake_minimum_required(VERSION 3.30)
set(SAMPLE_NAME "tinypt-wavefront")

add_executable(${SAMPLE_NAME} tinypt-wavefront.cu)
target_link_libraries(${SAMPLE_NAME} tinyobjloader)
target_include_directories(${SAMPLE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext)