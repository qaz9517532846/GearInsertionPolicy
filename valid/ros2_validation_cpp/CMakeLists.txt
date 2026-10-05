cmake_minimum_required(VERSION 3.16)
project(gear_insertion_ros2_cpp)

# C++20, not 17: with LibTorch 2.15 (dev/nightly) + GCC 13, building against
# <torch/script.h> under -std=c++17 fails inside ATen/core/ivalue_inl.h
# (a `std::remove_pointer_t<...>`-based template resolves to the wrong
# specialization and loses `element_type`). Confirmed by actually building
# this package: C++17 fails with that error, C++20 compiles clean. If you're
# on an older/stable LibTorch release this may no longer be necessary, but
# C++20 is harmless to keep either way.
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
# LibTorch is not an ament/ROS package: point CMAKE_PREFIX_PATH at your
# `libtorch/` (or an installed `torch` Python package's `share/cmake/Torch`)
# when invoking colcon, e.g.:
#   colcon build --packages-select gear_insertion_ros2_cpp \
#       --cmake-args -DCMAKE_PREFIX_PATH=/path/to/libtorch
find_package(Torch REQUIRED)

add_executable(policy_inference_node src/policy_inference_node.cpp)
target_include_directories(policy_inference_node PRIVATE
  include
  ${TORCH_INCLUDE_DIRS}
)
ament_target_dependencies(policy_inference_node rclcpp std_msgs sensor_msgs geometry_msgs)
target_link_libraries(policy_inference_node ${TORCH_LIBRARIES})
target_compile_options(policy_inference_node PRIVATE ${TORCH_CXX_FLAGS})
# LibTorch is not on the default loader search path: embed an RPATH to it so
# the installed executable finds libc10.so/libtorch.so without needing
# LD_LIBRARY_PATH set at runtime.
set_target_properties(policy_inference_node PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
)

add_executable(success_monitor_node src/success_monitor_node.cpp)
target_include_directories(success_monitor_node PRIVATE include)
ament_target_dependencies(success_monitor_node rclcpp std_msgs)

install(TARGETS
  policy_inference_node
  success_monitor_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
