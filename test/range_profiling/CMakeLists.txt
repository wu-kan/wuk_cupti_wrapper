project(wuk_cupti_wrapper_test_range_profiling LANGUAGES CUDA)

add_executable(${PROJECT_NAME} main.cu cupti_wrapper.cc)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/common)

find_package(CUDAToolkit REQUIRED)

target_link_libraries(${PROJECT_NAME} CUDA::cuda_driver CUDA::cupti_static)

add_test(
  NAME ${PROJECT_NAME}
  COMMAND $<TARGET_FILE:${PROJECT_NAME}>
)