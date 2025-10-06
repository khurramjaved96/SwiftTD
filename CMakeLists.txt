cmake_minimum_required(VERSION 3.21)

project(SwiftTD LANGUAGES CXX)

# Build settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

# pybind11 + Python (Development.Module ensures proper extension build flags)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# C++ core library (not installed; linked into the Python extension)
add_library(SwiftTD STATIC
        src/cpp/SwiftTD.cpp
)
target_include_directories(SwiftTD PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src/cpp
)

# Python extension module (top-level: import swift_td)
pybind11_add_module(swift_td
        src/cpp/pybind.cpp
)

target_link_libraries(swift_td PRIVATE SwiftTD)

# Let scikit-build-core place the extension correctly inside the wheel
# (use SKBUILD_PLATLIB_DIR instead of Python_SITEARCH)
install(TARGETS swift_td
        LIBRARY DESTINATION "${SKBUILD_PLATLIB_DIR}"
        ARCHIVE DESTINATION "${SKBUILD_PLATLIB_DIR}"
        RUNTIME DESTINATION "${SKBUILD_PLATLIB_DIR}"
)
