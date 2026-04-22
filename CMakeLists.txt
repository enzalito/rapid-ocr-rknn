cmake_minimum_required(VERSION 3.14)
project(rapid_ocr_rknn CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT "${CMAKE_SYSTEM_NAME}" MATCHES "^(Linux|Android)")
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()
set(PLATFORM "${CMAKE_SYSTEM_NAME}")

message(STATUS "Detected platform: ${CMAKE_SYSTEM_NAME}")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)")
    if ("${PLATFORM}" STREQUAL "Android")
        set(ARCH "arm64-v8a")
    elseif("${PLATFORM}" STREQUAL "Linux")
        set(ARCH "aarch64")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^arm")
    if ("${PLATFORM}" STREQUAL "Android")
        set(ARCH "armeabi-v7a")
    elseif("${PLATFORM}" STREQUAL "Linux")
        if (DEFINED UCLIBC)
            set(ARCH "armhf-uclib")
        else()
            set(ARCH "armhf")
        endif()
    endif()
endif()

if(NOT DEFINED ARCH)
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(DEFINED UCLIBC)
    if("${ARCH}" STREQUAL "armhf")
        set(ARCH "armhf-uclib")
    else()
        message(FATAL_ERROR "UCLIBC is not supported for ${PLATFORM} / ${ARCH}")
    endif()
endif()

message(STATUS "Detected architecture: ${ARCH}")

find_package(OpenCV REQUIRED)

add_library(rknn_api SHARED IMPORTED)
set_target_properties(rknn_api PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/vendor/rknn/${PLATFORM}/${ARCH}/librknnrt.so"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/vendor/rknn/${PLATFORM}/include"
)

file(GLOB SOURCES "src/*.cpp")
add_library(rapid_ocr_rknn ${SOURCES})
target_include_directories(rapid_ocr_rknn PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(rapid_ocr_rknn PUBLIC 
    ${OpenCV_LIBS}
    rknn_api
)

install(TARGETS rapid_ocr_rknn DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

install(FILES "vendor/rknn/${PLATFORM}/${ARCH}/librknnrt.so" DESTINATION lib)
install(DIRECTORY "vendor/rknn/${PLATFORM}/include/" DESTINATION include)
