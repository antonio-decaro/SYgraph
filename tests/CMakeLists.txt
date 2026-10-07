link_libraries(sygraph)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(bitmap_frontier frontier/bitmap_frontier.cpp)
add_executable(mlb_frontier frontier/mlb_frontier.cpp)
add_executable(mlb_semantics frontier/mlb_semantics.cpp)
add_executable(mlb_active_frontier_large frontier/mlb_active_frontier_large.cpp)
add_executable(bitmap_semantics frontier/bitmap_semantics.cpp)
add_executable(csr formats/csr.cpp)
add_executable(coo2csr_weighted formats/coo2csr.cpp)
add_executable(coo2csr_unweighted formats/coo2csr_unweighted.cpp)
add_executable(format_properties formats/properties.cpp)
add_executable(graph_build graph/graph.cpp)
add_executable(graph_teardown graph/graph_teardown.cpp)
add_executable(advance operators/advance.cpp)
add_executable(advance_graph operators/advance_graph.cpp)
add_executable(advance_pull operators/advance_pull.cpp)
add_executable(filter_compute operators/filter_compute.cpp)
add_executable(intersection_operator operators/intersection.cpp)
add_executable(bfs_algorithm algorithms/bfs.cpp)
add_executable(sssp_algorithm algorithms/sssp.cpp)
add_executable(cc_algorithm algorithms/cc.cpp)
add_executable(tc_algorithm algorithms/tc.cpp)
add_executable(bc_algorithm algorithms/bc.cpp)

get_directory_property(all_targets BUILDSYSTEM_TARGETS)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -fsycl-unnamed-lambda")
# Tests check results with assert(); keep them active in Release/RelWithDebInfo builds too.
foreach(test_target IN LISTS all_targets)
  target_compile_options(${test_target} PRIVATE -UNDEBUG)
endforeach()
if (DEFINED ARCH AND NOT ARCH STREQUAL "")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl-targets=${ARCH}")
endif()

enable_testing()

get_filename_component(SYGRAPH_TEST_COMPILER_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(SYGRAPH_TEST_ONEAPI_ROOT "${SYGRAPH_TEST_COMPILER_DIR}/.." ABSOLUTE)
set(SYGRAPH_TEST_ENV "")
if (EXISTS "${SYGRAPH_TEST_ONEAPI_ROOT}/lib")
  list(APPEND SYGRAPH_TEST_ENV "LD_LIBRARY_PATH=${SYGRAPH_TEST_ONEAPI_ROOT}/lib")
endif()

add_test(
  NAME test_bitmap_frontier
  COMMAND bitmap_frontier
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_mlb_frontier
  COMMAND mlb_frontier
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME mlb_semantics
  COMMAND mlb_semantics
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME mlb_active_frontier_large
  COMMAND mlb_active_frontier_large
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME bitmap_semantics
  COMMAND bitmap_semantics
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME test_csr_format
  COMMAND csr
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME coo2csr_weighted
  COMMAND coo2csr_weighted
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME coo2csr_unweighted
  COMMAND coo2csr_unweighted
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME format_properties
  COMMAND format_properties
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME graph_build
  COMMAND graph_build
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME graph_teardown
  COMMAND graph_teardown
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME advance_operator
  COMMAND advance
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME advance_graph_operator
  COMMAND advance_graph
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME advance_pull_operator
  COMMAND advance_pull
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME filter_compute_operator
  COMMAND filter_compute
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME intersection_operator
  COMMAND intersection_operator
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME bfs_algorithm
  COMMAND bfs_algorithm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME sssp_algorithm
  COMMAND sssp_algorithm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME cc_algorithm
  COMMAND cc_algorithm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME tc_algorithm
  COMMAND tc_algorithm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

add_test(
  NAME bc_algorithm
  COMMAND bc_algorithm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

set_tests_properties(
  test_bitmap_frontier
  test_mlb_frontier
  mlb_semantics
  mlb_active_frontier_large
  bitmap_semantics
  test_csr_format
  coo2csr_weighted
  coo2csr_unweighted
  format_properties
  graph_build
  graph_teardown
  advance_operator
  advance_graph_operator
  advance_pull_operator
  filter_compute_operator
  intersection_operator
  bfs_algorithm
  sssp_algorithm
  cc_algorithm
  tc_algorithm
  bc_algorithm
  PROPERTIES ENVIRONMENT "${SYGRAPH_TEST_ENV}"
)
