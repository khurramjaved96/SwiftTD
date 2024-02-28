cmake_minimum_required(VERSION 3.26)
project(trueonlinetdlambdatest)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)
# add other compiler optimizations=


set(ENV{PKG_CONFIG_PATH} "/Users/khurramjaved/miniconda3/lib/pkgconfig")
find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)
pkg_check_modules(mariadb REQUIRED IMPORTED_TARGET mariadb)

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
find_package(ZLIB REQUIRED)
find_package(Threads REQUIRED)


include_directories(${mariadb_INCLUDE_DIRS})

add_executable(Algorithm true_online_td_lambda.cpp src/utils.cpp)
add_executable(Count count.cpp)
add_subdirectory(src)

# link libraries list
set(link_libraries_list)
# append Exp to link_libraries_list
list(APPEND link_libraries_list Exp)
# append PkgConfig::mariadb to link_libraries_list
list(APPEND link_libraries_list PkgConfig::mariadb)
# append dl to link_libraries_list
list(APPEND link_libraries_list NN)
# append Threads::Threads to link_libraries_list
list(APPEND link_libraries_list Threads::Threads)
# append pthread to link_libraries_list
list(APPEND link_libraries_list ${OpenCV_LIBS}  )
list(APPEND link_libraries_list ZLIB::ZLIB)
list(APPEND link_libraries_list pthread)
list(APPEND link_libraries_list Env)
# append OpenCV_LIBS to link_libraries_list
#list(APPEND link_libraries_list ${OpenCV_LIBS})


target_link_libraries(Algorithm PRIVATE ${link_libraries_list})
target_link_libraries(Count PRIVATE Exp PkgConfig::mariadb)

