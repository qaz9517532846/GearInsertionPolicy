// SPDX-License-Identifier: BSD-3-Clause
//
// C++ (rclcpp + LibTorch) equivalent of
// ros2_validation/gear_insertion_ros2/policy_inference_node.py
//
// Loads the TorchScript policy exported by GearInsertionPolicy/scripts/play.py
// (policy.pt) via LibTorch's `torch::jit::load`, and runs the exact same
// closed loop as the Python node: subscribe joint_states + gear_shaft_pose,
// assemble the observation vector in the SAME order as
// ObservationsCfg.PolicyCfg (joint_pos, joint_vel, gear_shaft_pos,
// gear_shaft_quat_xyzw), run the policy, clamp to [-1, 1] (reproducing
// RSL-RL's clip_actions=1.0, which is NOT baked into the exported module --
// see isaaclab_rl/rsl_rl/exporter.py), and publish the raw action.
//
// Why a C++ version at all: this is the shape a real Isaac ROS / production
// inference node usually takes (rclcpp + LibTorch or TensorRT, no Python
// interpreter/GIL in the control loop). Swap this in wherever the Python node
// was used -- the ROS2 topic contract (see README.md) is identical, so the
// Isaac-Sim bridge node and success_monitor_node don't know or care which
// implementation of the "AI model" they're talking to.
//
// Threading note: this node relies on rclpy's/rclcpp's default
// SingleThreadedExecutor (used by `rclcpp::spin`) so that the subscription
// callbacks and the inference timer callback never run concurrently. If you
// move this to a MultiThreadedExecutor or a callback group that allows
// concurrent callbacks, the observation member variables below need a mutex.

#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gear_insertion_ros2_cpp/tiny_json.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "std_msgs/msg/string.hpp"
#include "torch/script.h"

using namespace std::chrono_literals;

namespace gear_insertion_ros2_cpp
{

class PolicyInferenceNode : public rclcpp::Node
{
public:
  PolicyInferenceNode()
  : Node("gear_insertion_policy_inference")
  {
    this->declare_parameter<std::string>("policy_path", "");
    this->declare_parameter<int>("num_arm_joints", 6);
    this->declare_parameter<double>("control_hz", 30.0);
    this->declare_parameter<std::string>("ros_namespace", "gear_insertion");
    this->declare_parameter<std::string>("device", "cpu");

    const std::string policy_path = this->get_parameter("policy_path").as_string();
    if (policy_path.empty()) {
      throw std::runtime_error(
              "The 'policy_path' parameter is required, e.g.:\n"
              "  ros2 run gear_insertion_ros2_cpp policy_inference_node "
              "--ros-args -p policy_path:=/path/to/policy.pt -p num_arm_joints:=6");
    }
    num_arm_joints_ = this->get_parameter("num_arm_joints").as_int();
    const double control_hz = this->get_parameter("control_hz").as_double();
    const std::string device_str = this->get_parameter("device").as_string();
    device_ = (device_str == "cuda") ? torch::kCUDA : torch::kCPU;

    const std::string ns = strip_slashes(
      this->get_parameter("ros_namespace").as_string().empty() ?
      "gear_insertion" : this->get_parameter("ros_namespace").as_string());

    RCLCPP_INFO(this->get_logger(), "Loading TorchScript policy from '%s' (device=%s)",
      policy_path.c_str(), device_str.c_str());
    policy_ = torch::jit::load(policy_path, device_);
    policy_.eval();

    joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/" + ns + "/joint_states", 10,
      std::bind(&PolicyInferenceNode::on_joint_states, this, std::placeholders::_1));
    gear_shaft_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/" + ns + "/gear_shaft_pose", 10,
      std::bind(&PolicyInferenceNode::on_gear_shaft_pose, this, std::placeholders::_1));
    episode_event_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/" + ns + "/episode_event", 10,
      std::bind(&PolicyInferenceNode::on_episode_event, this, std::placeholders::_1));
    action_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/" + ns + "/policy_action", 10);

    const auto period = std::chrono::duration<double>(1.0 / (control_hz > 0.0 ? control_hz : 30.0));
    timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&PolicyInferenceNode::on_control_tick, this));

    RCLCPP_INFO(this->get_logger(), "Policy inference node (C++) ready at %.1f Hz on namespace '/%s'.",
      control_hz, ns.c_str());
  }

private:
  static std::string strip_slashes(const std::string & s)
  {
    size_t start = s.find_first_not_of('/');
    size_t end = s.find_last_not_of('/');
    if (start == std::string::npos) {return s;}
    return s.substr(start, end - start + 1);
  }

  void on_joint_states(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    joint_pos_.assign(msg->position.begin(), msg->position.end());
    joint_vel_.assign(msg->velocity.begin(), msg->velocity.end());
  }

  void on_gear_shaft_pose(const geometry_msgs::msg::PoseStamped::SharedPtr msg)
  {
    // geometry_msgs fields are double; the observation vector (and the policy's
    // input tensor) is float32, matching the dtype used throughout training.
    gear_shaft_pos_ = {
      static_cast<float>(msg->pose.position.x),
      static_cast<float>(msg->pose.position.y),
      static_cast<float>(msg->pose.position.z)
    };
    gear_shaft_quat_xyzw_ = {
      static_cast<float>(msg->pose.orientation.x), static_cast<float>(msg->pose.orientation.y),
      static_cast<float>(msg->pose.orientation.z), static_cast<float>(msg->pose.orientation.w)
    };
    have_pose_ = true;
  }

  void on_episode_event(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!json_has_event(msg->data, "episode_end")) {
      return;
    }
    // Clear the exported policy's recurrent (LSTM) hidden/cell state -- see
    // isaaclab_rl/rsl_rl/exporter.py `reset_memory()`, exposed via
    // `@torch.jit.export def reset(self): ...`. LibTorch calls a non-forward
    // exported method through `get_method(name)`.
    policy_.get_method("reset")({});
    RCLCPP_INFO(this->get_logger(), "episode_end received -> policy hidden state reset.");
  }

  void on_control_tick()
  {
    if (!have_pose_ || static_cast<int>(joint_pos_.size()) != num_arm_joints_ ||
      static_cast<int>(joint_vel_.size()) != num_arm_joints_)
    {
      return;  // haven't received a full observation yet
    }

    std::vector<float> obs;
    obs.reserve(2 * num_arm_joints_ + 7);
    obs.insert(obs.end(), joint_pos_.begin(), joint_pos_.end());
    obs.insert(obs.end(), joint_vel_.begin(), joint_vel_.end());
    obs.insert(obs.end(), gear_shaft_pos_.begin(), gear_shaft_pos_.end());
    obs.insert(obs.end(), gear_shaft_quat_xyzw_.begin(), gear_shaft_quat_xyzw_.end());

    torch::Tensor obs_tensor = torch::from_blob(
      obs.data(), {1, static_cast<long>(obs.size())}, torch::kFloat32).clone().to(device_);

    torch::Tensor raw_action;
    {
      torch::NoGradGuard no_grad;
      std::vector<torch::jit::IValue> inputs{obs_tensor};
      raw_action = policy_.forward(inputs).toTensor();
    }
    torch::Tensor action = torch::clamp(raw_action, -1.0, 1.0).to(torch::kCPU).squeeze(0).contiguous();

    std_msgs::msg::Float32MultiArray msg;
    const float * action_data = action.data_ptr<float>();
    msg.data.assign(action_data, action_data + action.numel());
    action_pub_->publish(msg);
  }

  torch::jit::script::Module policy_;
  torch::Device device_ = torch::kCPU;
  int num_arm_joints_ = 6;

  std::vector<float> joint_pos_;
  std::vector<float> joint_vel_;
  std::array<float, 3> gear_shaft_pos_{};
  std::array<float, 4> gear_shaft_quat_xyzw_{};
  bool have_pose_ = false;

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr gear_shaft_pose_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr episode_event_sub_;
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr action_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace gear_insertion_ros2_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<gear_insertion_ros2_cpp::PolicyInferenceNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("gear_insertion_policy_inference"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
