cmake_minimum_required(VERSION 2.8)
project( mycv )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( mycv camcalib.cpp )
target_link_libraries( mycv ${OpenCV_LIBS} )
SET(CMAKE_BUILD_TYPE "Debug")
