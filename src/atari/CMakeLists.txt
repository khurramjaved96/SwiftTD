# Include src/ and cmake binary directory (for version.hpp)
#include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

#configure_file ("version.hpp" "version.hpp")
add_subdirectory(common)
add_subdirectory(emucore)
add_subdirectory(environment)
add_subdirectory(games)
target_sources(Atari PRIVATE ale_interface.hpp ale_interface.cpp)
