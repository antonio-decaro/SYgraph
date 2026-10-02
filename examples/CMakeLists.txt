# Copyright (c) 2025 University of Salerno
# SPDX-License-Identifier: Apache-2.0

include(FetchContent)

set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.1
  GIT_SHALLOW TRUE
)

FetchContent_GetProperties(cli11)
if(NOT cli11_POPULATED)
  FetchContent_Populate(cli11)
  add_subdirectory(${cli11_SOURCE_DIR} ${cli11_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

link_libraries(sygraph CLI11::CLI11)

add_executable(bfs bfs/bfs.cpp)
add_executable(sssp sssp/sssp.cpp)
add_executable(tc tc/tc.cpp)
add_executable(bc bc/bc.cpp)
add_executable(cc cc/cc.cpp)

set_property(DIRECTORY ${CMAKE_SOURCE_DIR}/ PROPERTY CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

# Define the GRAPH_LOCATION parameter
set(GRAPH_LOCATION "device" CACHE STRING "Location of the graph: host, device, or shared")
set_property(CACHE GRAPH_LOCATION PROPERTY STRINGS "host" "device" "shared")

# Map GRAPH_LOCATION to integers in compile definitions
if (GRAPH_LOCATION STREQUAL "host")
    message(WARNING "GRAPH_LOCATION=host is not recommended for performance. Use at your own risk.")
    add_compile_definitions(GRAPH_LOCATION=0)
elseif (GRAPH_LOCATION STREQUAL "device")
    add_compile_definitions(GRAPH_LOCATION=1)
elseif (GRAPH_LOCATION STREQUAL "shared")
    add_compile_definitions(GRAPH_LOCATION=2)
else()
    message(FATAL_ERROR "Invalid GRAPH_LOCATION value. Must be 'host', 'device', or 'shared'.")
endif()

# If compiling with oneAPI
if (CMAKE_CXX_COMPILER MATCHES ".*dpcpp.*" OR CMAKE_CXX_COMPILER MATCHES ".*clang.*" OR CMAKE_CXX_COMPILER MATCHES ".*icpx.*")
  if (DEFINED ARCH AND NOT ARCH STREQUAL "")
    message("AOT Compilation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${ARCH}")
  else()
    message("JIT Compilation")
    # Identify the available targets

    # Check for CUDA
    find_package(CUDAToolkit QUIET)

    # Check for OpenCL
    find_package(OpenCL QUIET)

    # Check for HIP
    find_package(hip QUIET)

    # Initialize an empty list for SYCL targets
    set(SYCL_TARGETS "")

    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found. Adding 'nvptx64-nvidia-cuda' to SYCL targets.")
        list(APPEND SYCL_TARGETS "nvptx64-nvidia-cuda")
    endif()

    if (OpenCL_FOUND)
        message(STATUS "Intel OpenCL driver found. Adding 'spir64' to SYCL targets.")
        list(APPEND SYCL_TARGETS "spir64")
    endif()

    if (hip_FOUND)
        message(STATUS "HIP found. Adding 'amdgcn-amd-amdhsa' to SYCL targets.")
        list(APPEND SYCL_TARGETS "amdgcn-amd-amdhsa")
    endif()

    # Remove duplicate entries from SYCL_TARGETS
    list(REMOVE_DUPLICATES SYCL_TARGETS)

    # Convert SYCL_TARGETS list to a semicolon-separated string
    string(REPLACE ";" "," SYCL_TARGETS_STR "${SYCL_TARGETS}")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-targets=${SYCL_TARGETS_STR}")
  endif()
endif()
