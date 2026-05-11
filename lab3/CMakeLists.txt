cmake_minimum_required(VERSION 3.12)
project("lab3" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


find_package(MPI REQUIRED)

add_executable(lab3 "main.cpp")

target_link_libraries(lab3 PRIVATE MPI::MPI_CXX)

if(CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
   CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(lab3 PRIVATE "-Wall" "-Wextra" "-pedantic")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(lab3 PRIVATE "/W4" "/utf-8")
endif()