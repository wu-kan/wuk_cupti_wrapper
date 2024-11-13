cmake_minimum_required(VERSION 3.27)

project(wuk_cupti_wrapper LANGUAGES CXX)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/wuk TYPE INCLUDE)

add_subdirectory(lib)

include(CTest)

add_subdirectory(test)
