add_subdirectory(supported)
target_sources(Atari PRIVATE
  PRIVATE
    Roms.cpp
    RomSettings.cpp
    RomUtils.cpp
)
