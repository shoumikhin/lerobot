cmake_minimum_required(VERSION 3.20)
project(lerobot_runner CXX)

set(CMAKE_CXX_STANDARD 17)

# -- libtorch (must match the torch version used for AOTI export) --
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# -- Bundle directory (set via -DBUNDLE_DIR=/path/to/bundle) --
if(NOT DEFINED BUNDLE_DIR)
    set(BUNDLE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../example_bundle")
endif()
add_compile_definitions(BUNDLE_DIR="${BUNDLE_DIR}")

add_executable(lerobot_runner runner.cpp)

target_link_libraries(lerobot_runner "${TORCH_LIBRARIES}")
target_include_directories(lerobot_runner PRIVATE "${TORCH_INCLUDE_DIRS}")

# torch AOTI loader requires these flags on Linux.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(lerobot_runner dl)
    set_target_properties(lerobot_runner PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endif()

message(STATUS "Bundle dir: ${BUNDLE_DIR}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
