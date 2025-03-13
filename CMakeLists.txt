cmake_minimum_required(VERSION 3.21)
project(SwiftTD)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Create the main library
add_library(SwiftTD STATIC SwiftTD.cpp)
target_include_directories(SwiftTD PUBLIC ${CMAKE_SOURCE_DIR})

# Create Python module
pybind11_add_module(swift_td pybind.cpp)
target_link_libraries(swift_td PRIVATE SwiftTD)

# Install targets
install(TARGETS SwiftTD
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(FILES SwiftTD.h DESTINATION include)

install(TARGETS swift_td 
        LIBRARY DESTINATION ${Python_SITEARCH}
        ARCHIVE DESTINATION ${Python_SITEARCH})