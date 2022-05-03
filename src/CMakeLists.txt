cmake_minimum_required (VERSION 2.8)

file(GLOB_RECURSE CFILES "${PROJECT_SOURCE_DIR}/*.cpp")
SET_SOURCE_FILES_PROPERTIES(${CFILES} PROPERTIES LANGUAGE CXX )

aux_source_directory(. BruckVariants_SOURCES)
add_library(bruckLib ${BruckVariants_SOURCES})

target_link_libraries(bruckLib ${MPI_C_LIBRARIES})
