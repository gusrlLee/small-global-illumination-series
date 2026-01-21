cmake_minimum_required(VERSION 3.30)
set(SAMPLE_NAME "tinypt-optix")

find_package(CUDAToolkit REQUIRED)

if(NOT DEFINED OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0" CACHE PATH "Path to OptiX SDK")
endif()

message(STATUS "Using OptiX SDK at: ${OptiX_INSTALL_DIR}")

add_executable(${SAMPLE_NAME} tinypt-optix.cu)

target_link_libraries(${SAMPLE_NAME} tinyobjloader)
target_link_libraries(${SAMPLE_NAME} CUDA::cudart)

target_include_directories(${SAMPLE_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/ext)
target_include_directories(${SAMPLE_NAME} PUBLIC ${OptiX_INSTALL_DIR}/include)
target_include_directories(${SAMPLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(compile_cuda_to_ptx target_name input_file output_ptx)
    add_custom_command(
        OUTPUT ${output_ptx}
        COMMAND ${CMAKE_CUDA_COMPILER}
        -ptx 
        -I${OptiX_INSTALL_DIR}/include
        -I${CMAKE_SOURCE_DIR}/ext
        -I${CMAKE_CURRENT_SOURCE_DIR}
        --use_fast_math
        -lineinfo
        --keep-device-functions
        -o ${output_ptx} 
        ${input_file}
        DEPENDS ${input_file}
        COMMENT "Compiling CUDA to PTX: ${input_file}"
    )
    add_custom_target(${target_name}_ptx DEPENDS ${output_ptx})
    add_dependencies(${SAMPLE_NAME} ${target_name}_ptx)
endfunction()