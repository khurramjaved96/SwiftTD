cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(SwiftTD)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3)

add_executable(Test main.cpp)
add_library(SwiftTD SwiftTD.cpp)

target_link_libraries(Test PRIVATE  SwiftTD)