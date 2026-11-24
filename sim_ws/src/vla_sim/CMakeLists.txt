cmake_minimum_required(VERSION 3.0.2)
project(vla_sim)
find_package(catkin REQUIRED COMPONENTS roscpp rospy geometry_msgs nav_msgs sensor_msgs std_msgs visualization_msgs quadrotor_msgs gazebo_msgs tf traj_utils traj_opt plan_env path_searching cv_bridge)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(gazebo REQUIRED)
catkin_package()
include_directories(include ${catkin_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
add_library(vla_acquisition_lidar SHARED src/acquisition_lidar_plugin.cpp)
target_compile_features(vla_acquisition_lidar PRIVATE cxx_std_14)
target_include_directories(vla_acquisition_lidar SYSTEM PRIVATE ${GAZEBO_INCLUDE_DIRS})
target_link_libraries(vla_acquisition_lidar ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(vla_acquisition_lidar ${catkin_EXPORTED_TARGETS})
add_executable(vla_cloud_bridge src/cloud_bridge.cpp)
target_compile_features(vla_cloud_bridge PRIVATE cxx_std_14)
target_link_libraries(vla_cloud_bridge ${catkin_LIBRARIES})
add_dependencies(vla_cloud_bridge ${catkin_EXPORTED_TARGETS})
add_executable(vla_traj_server src/traj_server.cpp)
target_compile_features(vla_traj_server PRIVATE cxx_std_14)
target_link_libraries(vla_traj_server ${catkin_LIBRARIES})
add_dependencies(vla_traj_server ${catkin_EXPORTED_TARGETS})
# Compile the original FSM/node against a minimal simulation-only manager overlay.
# The original checkout is read-only input; no source patch is applied there.
find_path(VLA_BASE_PLANNER src/diff_planner_node.cpp HINTS
  "${CMAKE_CURRENT_SOURCE_DIR}/../plan_manage"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../Diff-planner/src/diff_planner/plan_manage")
if(NOT VLA_BASE_PLANNER)
  message(FATAL_ERROR "Original Diff-Planner source package is required for the simulation overlay")
endif()
add_executable(vla_diff_planner_node
  "${VLA_BASE_PLANNER}/src/diff_planner_node.cpp"
  src/diff_replan_fsm.cpp
  src/planner_manager.cpp)
target_include_directories(vla_diff_planner_node BEFORE PRIVATE
  "${VLA_BASE_PLANNER}/include")
target_compile_features(vla_diff_planner_node PRIVATE cxx_std_14)
target_link_libraries(vla_diff_planner_node ${catkin_LIBRARIES})
add_dependencies(vla_diff_planner_node ${catkin_EXPORTED_TARGETS})
catkin_install_python(PROGRAMS scripts/flight_controller.py scripts/scene_visualization.py scripts/smoke_test.py scripts/record_demo.py scripts/camera_smoke_test.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
# Importable guard module must be copied, not wrapped in a catkin script relay.
configure_file(scripts/photo_guard.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/photo_guard.py COPYONLY)
configure_file(scripts/photo_search.py ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/photo_search.py COPYONLY)
install(FILES scripts/photo_guard.py scripts/photo_search.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS vla_cloud_bridge vla_traj_server vla_diff_planner_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(TARGETS vla_acquisition_lidar LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION})
install(DIRECTORY launch config worlds models rviz DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
