add_subdirectory(supported)
target_sources(Algorithm
  PRIVATE
    Roms.cpp
    RomSettings.cpp
    RomUtils.cpp
)


target_sources(TDLambda
        PRIVATE
        Roms.cpp
        RomSettings.cpp
        RomUtils.cpp
)
