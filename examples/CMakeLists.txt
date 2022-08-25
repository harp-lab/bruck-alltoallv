cmake_minimum_required (VERSION 2.8)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/examples/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX COMPILE_FLAGS "-std=c++11")

add_executable(ubruck ./uniform_bruck_example.cpp)
add_executable(nubruck ./non_uniform_bruck_example.cpp)
add_executable(rbruck ./radix_r_bruck_example.cpp)

target_link_libraries(ubruck bruckLib)
target_link_libraries(nubruck bruckLib)
target_link_libraries(rbruck bruckLib)