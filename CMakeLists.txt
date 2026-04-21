cmake_minimum_required(VERSION 3.14)
project(rapid_ocr_rknn CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT DEFINED PLATFORM)
    set(PLATFORM "Linux")
    message(STATUS "PLATFORM not defined, defaulting to ${PLATFORM}")
endif ()
if (NOT DEFINED ARCH)
    set(ARCH "aarch64")
    message(STATUS "ARCH not defined, defaulting to ${ARCH}")
endif ()

find_package(OpenCV REQUIRED)

add_library(rknn_api SHARED IMPORTED)
set_target_properties(rknn_api PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/vendor/rknn/${PLATFORM}/${ARCH}/librknnrt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/vendor/rknn/${PLATFORM}/include"
)

file(GLOB SOURCES "src/*.cpp")
add_library(rapid_ocr_rknn STATIC ${SOURCES})
target_include_directories(rapid_ocr_rknn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(rapid_ocr_rknn PUBLIC 
    ${OpenCV_LIBS}
    rknn_api
)
