# ------------------------------------------------------------------------
#  Gunrock: Sub-Project PageRank
# ------------------------------------------------------------------------
project(sgc)
message("-- Project Added: ${PROJECT_NAME}")
add_library(libcholmod SHARED IMPORTED)
set_property(TARGET libcholmod PROPERTY IMPORTED_LOCATION "/home/husong/SuiteSparse-5.7.1/lib/libcholmod.so")
include_directories("/home/husong/SuiteSparse-5.7.1/include")
include(${CMAKE_SOURCE_DIR}/cmake/SetSubProject.cmake)
target_link_libraries(${PROJECT_NAME} libcholmod)