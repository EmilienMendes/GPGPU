cmake_minimum_required(VERSION 3.16)
project(ProjetGPU)

# On active le support de CUDA
enable_language(CUDA)

set(SOURCES
        "src/main.cu"
        "src/view_test/student_view_naive.cu"
        "src/view_test/student_view_optimized.cu"
        "src/tiled/student_tiled_naive.cu"
        "src/tiled/student_tiled_optimized.cu"
        "src/cpu/reference.cpp"
        "utils/geometry.cpp"
        "utils/chronoGPU.cu"
        "utils/chronoCPU.cpp"
        "utils/ppm.cpp"
        "utils/variable.hpp"
        "src/view_test/student_view_double_kernel.cu"
        "src/view_test/student_view_fully_optimized.cu"

)

set (VARIABLES
        "-c"
        10,
        10)
set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(ProjetGPU ${SOURCES} )
target_compile_features(ProjetGPU PRIVATE cxx_std_17)

set_target_properties(ProjetGPU PROPERTIES CUDA_ARCHITECTURES "70")
set_target_properties(ProjetGPU PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
