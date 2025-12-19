cmake_minimum_required(VERSION 3.5)
project(r2d2)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
# Force OpenCV 4.5.4
set(OpenCV_DIR "/home/andreagrandi/build")
find_package(OpenCV 4.5.4 EXACT REQUIRED)
message(STATUS "Using OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

find_package(ament_cmake REQUIRED)
# Find cv_bridge from local workspace (not from /opt/ros/foxy)
# Add local install directory to CMAKE_PREFIX_PATH with higher priority
list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../install")
find_package(cv_bridge REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(tracker src/tracker.cpp)
ament_target_dependencies(tracker rclcpp cv_bridge geometry_msgs sensor_msgs std_msgs)
target_link_libraries(tracker ${OpenCV_LIBS})
# Set RPATH to use OpenCV from /home/andreagrandi/build/lib
set_target_properties(tracker PROPERTIES
  INSTALL_RPATH "/home/andreagrandi/build/lib"
  BUILD_WITH_INSTALL_RPATH TRUE
)

#if(BUILD_TESTING)
#  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
#  ament_lint_auto_find_test_dependencies()
#endif()

install (TARGETS
  tracker
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY config description launch worlds maps
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
