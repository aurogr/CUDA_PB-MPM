cmake_minimum_required(VERSION 3.18)
project(CUDA_MPM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Enforce static runtime linking for both CUDA and MSVC C++ runtime
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Target host GPU architecture (native builds for your installed GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# Gather all source files using absolute paths and CONFIGURE_DEPENDS
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/solver.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/grid.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/particleSystem.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/constants.h"
)

add_executable(${PROJECT_NAME} ${SRC_FILES})

# Find packages
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Link dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE 
    glfw               
    OpenGL::GL          
    CUDA::cudart        
)

# Include directory so headers resolve nicely
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Compiler options for CUDA host/device integration
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)