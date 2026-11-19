cmake_minimum_required(VERSION 3.24)
project(lettuce LANGUAGES CXX)

# Speed is the point of this project, so an unspecified build type means Release.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_EXTENSIONS OFF)
# For clangd and CLion; harmless elsewhere.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(LETTUCE_CUDA AUTO CACHE STRING "Build the CUDA backend: AUTO, ON or OFF")
set_property(CACHE LETTUCE_CUDA PROPERTY STRINGS AUTO ON OFF)
# x86-64-v3 (AVX2/FMA) runs on every H100 host CPU and lets the CPU backend vectorise. Use
# "native" for a single machine, or empty for the compiler default.
set(LETTUCE_CPU_ARCH "x86-64-v3" CACHE STRING "Value for -march on x86-64 (empty = compiler default)")
# LLVM's libc++ with Clang: Clang 18 cannot compile libstdc++ 13's <expected>, and libc++ is the
# LLVM toolchain's own library. Off for GCC/nvcc builds.
option(LETTUCE_LIBCXX "Use libc++ (Clang only)" OFF)
option(LETTUCE_LLD "Link with lld" OFF)

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(lettuce_use_cuda OFF)
if(NOT LETTUCE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # H100 is sm_90. Pass -DCMAKE_CUDA_ARCHITECTURES="80;90" to also cover A100 nodes.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
      set(CMAKE_CUDA_ARCHITECTURES 90)
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Clang compiles CUDA in the same dialect as the host code; nvcc 12 tops out at C++20, so the
    # .cu sources and the headers they include must stay C++20-clean.
    if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
      set(CMAKE_CUDA_STANDARD 23)
    else()
      set(CMAKE_CUDA_STANDARD 20)
    endif()
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(lettuce_use_cuda ON)
  elseif(LETTUCE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "LETTUCE_CUDA=ON but no CUDA compiler was found (set CMAKE_CUDA_COMPILER or load the CUDA module)")
  endif()
endif()
message(STATUS "lettuce: CUDA backend ${lettuce_use_cuda}")

add_library(lettuce_core STATIC
  src/config.cpp
  src/scenarios.cpp
  src/snapshot.cpp
  src/render.cpp
  src/cpu_backend.cpp
  src/simulation.cpp
  src/live.cpp
  src/stb_impl.cpp
)
target_include_directories(lettuce_core PUBLIC src)
target_include_directories(lettuce_core SYSTEM PRIVATE vendor)
target_link_libraries(lettuce_core PUBLIC OpenMP::OpenMP_CXX)

if(lettuce_use_cuda)
  target_sources(lettuce_core PRIVATE src/cuda_backend.cu)
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lettuce_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-ffast-math -fcuda-flush-denormals-to-zero -gline-tables-only>)
  else()
    target_compile_options(lettuce_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo --expt-relaxed-constexpr>)
  endif()
  target_link_libraries(lettuce_core PUBLIC CUDA::cudart)
else()
  target_sources(lettuce_core PRIVATE src/cuda_stub.cpp)
endif()

if(LETTUCE_CPU_ARCH AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(lettuce_core PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-march=${LETTUCE_CPU_ARCH}>)
endif()
if(LETTUCE_LIBCXX)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "LETTUCE_LIBCXX needs Clang")
  endif()
  target_compile_options(lettuce_core PUBLIC $<$<COMPILE_LANGUAGE:CXX,CUDA>:-stdlib=libc++>)
  target_link_options(lettuce_core PUBLIC -stdlib=libc++)
  # CUDA 12.0's Thrust calls std::terminate without including <exception>; libstdc++ pulls it in
  # transitively, libc++ does not.
  if(lettuce_use_cuda AND CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    target_compile_options(lettuce_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-include exception>)
  endif()
endif()
if(LETTUCE_LLD)
  target_link_options(lettuce_core PUBLIC -fuse-ld=lld)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lettuce_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

add_executable(lettuce src/main.cpp)
target_link_libraries(lettuce PRIVATE lettuce_core)
add_executable(lettuce-bench src/bench.cpp)
target_link_libraries(lettuce-bench PRIVATE lettuce_core)
add_executable(lettuce-render src/render_main.cpp)
target_link_libraries(lettuce-render PRIVATE lettuce_core)

enable_testing()
file(GLOB lettuce_test_sources CONFIGURE_DEPENDS tests/*.cpp)
add_executable(lettuce-tests ${lettuce_test_sources})
target_link_libraries(lettuce-tests PRIVATE lettuce_core)
add_test(NAME lettuce-tests COMMAND lettuce-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
