cmake_minimum_required(VERSION 3.12)

add_definitions(-DWITH_NEON)
#add_definitions(-DWITH_AVX2)

set(CMAKE_CXX_STANDARD 11)

project(matrix) 

ADD_EXECUTABLE(main Main.cpp)
ADD_LIBRARY(Matrix matrix.c)

set(CMAKE_BUILD_TYPE "RELEASE")

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP found.")
    target_link_libraries(main PUBLIC Matrix OpenMP::OpenMP_CXX)
endif()
