cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 80)

set(FT "0" CACHE STRING "Description of FT")
# set(FT $ENV{FT} CACHE STRING "Compile-time parameter FT")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/include)
include_directories(${CMAKE_SOURCE_DIR}/cuda-samples/Common)
include_directories(${CMAKE_SOURCE_DIR})

# CUDA paths
set(CUDA_PATH "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda")
set(MATH_LIB "/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/math_libs/12.2")

# Library directories
link_directories(${MATH_LIB}/lib64)

# Specify the executable targets
# add_executable(main src/main.cu)
# target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# target_link_libraries(main PRIVATE cublas)

# add_executable(test_tf32_tensorop src/test_tf32_tensorop.cu)
# target_compile_options(test_tf32_tensorop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# target_compile_definitions(test_tf32_tensorop PRIVATE FT=${FT})
# target_link_libraries(test_tf32_tensorop PRIVATE cublas)

# add_executable(test_fp64_tensorop src/test_fp64_tensorop.cu)
# target_compile_options(test_fp64_tensorop PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# target_compile_definitions(test_fp64_tensorop PRIVATE FT=${FT})
# target_link_libraries(test_fp64_tensorop PRIVATE cublas)

add_executable(test_kmeans_tuner_fp32_fixK src/test_kmeans_tuner_fp32_fixK.cu)
target_compile_options(test_kmeans_tuner_fp32_fixK PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(test_kmeans_tuner_fp32_fixK PRIVATE FT=${FT})
target_link_libraries(test_kmeans_tuner_fp32_fixK PRIVATE cublas)

add_executable(test_kmeans_tuner_fp64_fixK src/test_kmeans_tuner_fp64_fixK.cu)
target_compile_options(test_kmeans_tuner_fp64_fixK PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(test_kmeans_tuner_fp64_fixK PRIVATE FT=${FT})
target_link_libraries(test_kmeans_tuner_fp64_fixK PRIVATE cublas)


add_executable(test_kmeans_tuner_fp32_fixN src/test_kmeans_tuner_fp32_fixN.cu)
target_compile_options(test_kmeans_tuner_fp32_fixN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(test_kmeans_tuner_fp32_fixN PRIVATE FT=${FT})
target_link_libraries(test_kmeans_tuner_fp32_fixN PRIVATE cublas)

add_executable(test_kmeans_tuner_fp64_fixN src/test_kmeans_tuner_fp64_fixN.cu)
target_compile_options(test_kmeans_tuner_fp64_fixN PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
target_compile_definitions(test_kmeans_tuner_fp64_fixN PRIVATE FT=${FT})
target_link_libraries(test_kmeans_tuner_fp64_fixN PRIVATE cublas)


# add_executable(test_cublas src/test_cublas.cu)
# target_compile_options(test_cublas PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
# target_link_libraries(test_cublas PRIVATE cublas)

# Clean the build
add_custom_target(clean-all
   COMMAND ${CMAKE_COMMAND} -P ${CMAKE_SOURCE_DIR}/CleanAll.cmake
)

# Note: You might need to create a CleanAll.cmake file or an alternative way to clean your build files, as CMake's clean command does not remove the executables.