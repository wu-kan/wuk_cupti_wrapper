project(wuk_cupti_wrapper_test_range_profiling LANGUAGES CUDA)

add_executable(${PROJECT_NAME} main.cu)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include)

target_include_directories(
  ${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/common)

target_link_libraries(${PROJECT_NAME} wuk_cupti_wrapper)

add_test(
  NAME ${PROJECT_NAME}
  COMMAND $<TARGET_FILE:${PROJECT_NAME}>
)