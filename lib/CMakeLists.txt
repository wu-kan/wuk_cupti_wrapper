cmake_minimum_required(VERSION 3.27)

project(wuk_cupti_wrapper LANGUAGES CXX)

add_library(${PROJECT_NAME} cupti_wrapper.cc)

install(TARGETS ${PROJECT_NAME})

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/wuk TYPE INCLUDE)

find_package(CUDAToolkit REQUIRED)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CUDAToolkit_LIBRARY_ROOT}/extras/CUPTI/include)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/extensions/include)

target_link_libraries(${PROJECT_NAME} CUDA::cuda_driver CUDA::cupti_static
                      CUDA::nvperf_host_static)

add_subdirectory(extensions)

target_link_libraries(${PROJECT_NAME} wuk_cupti_wrapper_ensions)