"""Launch the C++ (rclcpp + LibTorch) half of the Gear Insertion validation loop.

Drop-in replacement for gear_insertion_ros2's
gear_insertion_validation.launch.py -- same parameters, same topics. Does NOT
start the Isaac-Sim robot bridge; launch that separately with `python`
(see README.md).

Example:

    ros2 launch gear_insertion_ros2_cpp gear_insertion_validation_cpp.launch.py \\
        policy_path:=/abs/path/to/logs/rsl_rl/gear_assembly_ur10e/<run>/policy.pt \\
        num_arm_joints:=6
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    policy_path_arg = DeclareLaunchArgument(
        "policy_path", description="Absolute path to the exported policy.pt (from scripts/play.py)."
    )
    num_arm_joints_arg = DeclareLaunchArgument(
        "num_arm_joints", default_value="6", description="6 for UR10e, 7 for Flexiv Rizon 4s."
    )
    control_hz_arg = DeclareLaunchArgument("control_hz", default_value="30.0")
    ros_namespace_arg = DeclareLaunchArgument(
        "ros_namespace",
        default_value="gear_insertion",
        description="Must match --ros_namespace passed to scripts/ros2_isaac_sim_bridge.py.",
    )
    device_arg = DeclareLaunchArgument("device", default_value="cpu", description="cpu or cuda.")
    report_path_arg = DeclareLaunchArgument("report_path", default_value="")
    report_every_n_arg = DeclareLaunchArgument("report_every_n", default_value="1")

    policy_inference_node = Node(
        package="gear_insertion_ros2_cpp",
        executable="policy_inference_node",
        name="gear_insertion_policy_inference",
        output="screen",
        parameters=[
            {
                "policy_path": LaunchConfiguration("policy_path"),
                "num_arm_joints": LaunchConfiguration("num_arm_joints"),
                "control_hz": LaunchConfiguration("control_hz"),
                "ros_namespace": LaunchConfiguration("ros_namespace"),
                "device": LaunchConfiguration("device"),
            }
        ],
    )

    success_monitor_node = Node(
        package="gear_insertion_ros2_cpp",
        executable="success_monitor_node",
        name="gear_insertion_success_monitor",
        output="screen",
        parameters=[
            {
                "ros_namespace": LaunchConfiguration("ros_namespace"),
                "report_path": LaunchConfiguration("report_path"),
                "report_every_n": LaunchConfiguration("report_every_n"),
            }
        ],
    )

    return LaunchDescription(
        [
            policy_path_arg,
            num_arm_joints_arg,
            control_hz_arg,
            ros_namespace_arg,
            device_arg,
            report_path_arg,
            report_every_n_arg,
            policy_inference_node,
            success_monitor_node,
        ]
    )
