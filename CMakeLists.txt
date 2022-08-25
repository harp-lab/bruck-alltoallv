cmake_minimum_required (VERSION 2.8)

project (BruckVariants)
set (CMAKE_CXX_STANDARD 11)

message("Configuring MPR with MPI support")
find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_SOURCE_DIR}/examples")
include_directories("${PROJECT_SOURCE_DIR}/test")

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(test)

