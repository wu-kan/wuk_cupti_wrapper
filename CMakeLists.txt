cmake_minimum_required(VERSION 3.27)

project(wuk_cupti_wrapper)

add_subdirectory(lib)

include(CTest)

add_subdirectory(test)
