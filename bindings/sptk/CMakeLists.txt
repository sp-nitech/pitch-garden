cmake_minimum_required(VERSION 3.26...3.27)
project(pitch_garden_sptk LANGUAGES CXX)

find_package(
  Python 3.12
  COMPONENTS Interpreter Development.Module ${SKBUILD_SABI_COMPONENT}
  REQUIRED)
find_package(nanobind CONFIG REQUIRED)

# Build only the static library of SPTK.
set(SPTK_INSTALL
    OFF
    CACHE BOOL "" FORCE)
set(SPTK_INSTALL_DRAW_COMMANDS
    OFF
    CACHE BOOL "" FORCE)
set(SPTK_BUILD_COMMANDS
    OFF
    CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third_party/SPTK EXCLUDE_FROM_ALL)

# Use the stable ABI so that one wheel works with Python 3.12 and later.
nanobind_add_module(_core STABLE_ABI src/binding.cc)
target_compile_features(_core PRIVATE cxx_std_17)
target_link_libraries(_core PRIVATE sptk)
install(TARGETS _core LIBRARY DESTINATION pitch_garden_sptk)
