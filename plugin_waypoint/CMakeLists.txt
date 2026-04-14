cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(waypoint_follower)

find_package(gazebo REQUIRED)

include_directories(
  ${GAZEBO_INCLUDE_DIRS}
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

add_library(waypoint_follower SHARED WaypointFollower.cpp)
target_link_libraries(waypoint_follower ${GAZEBO_LIBRARIES})

# Output plugin as: libwaypoint_follower.so
set_target_properties(waypoint_follower PROPERTIES OUTPUT_NAME "waypoint_follower")