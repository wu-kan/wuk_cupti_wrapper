cmake_minimum_required(VERSION 3.27)

project(wuk_cupti_wrapper_ensions LANGUAGES CXX)

add_library(${PROJECT_NAME} src/profilerhost_util/Metric.cpp src/profilerhost_util/List.cpp src/profilerhost_util/Eval.cpp)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/c_util)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/profilerhost_util)
