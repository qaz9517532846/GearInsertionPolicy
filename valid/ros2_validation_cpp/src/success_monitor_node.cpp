// SPDX-License-Identifier: BSD-3-Clause
//
// C++ (rclcpp) equivalent of
// ros2_validation/gear_insertion_ros2/success_monitor_node.py
//
// Subscribes to /<ns>/episode_event and tallies a running success rate --
// see the Python node's docstring and README.md for the success/failure
// definition and its caveats (coarse proxy, not a rigorous insertion check).

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <string>

#include "gear_insertion_ros2_cpp/tiny_json.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

namespace gear_insertion_ros2_cpp
{

class SuccessMonitorNode : public rclcpp::Node
{
public:
  SuccessMonitorNode()
  : Node("gear_insertion_success_monitor")
  {
    this->declare_parameter<std::string>("ros_namespace", "gear_insertion");
    this->declare_parameter<int>("report_every_n", 1);
    this->declare_parameter<std::string>("report_path", "");

    const std::string ns = strip_slashes(
      this->get_parameter("ros_namespace").as_string().empty() ?
      "gear_insertion" : this->get_parameter("ros_namespace").as_string());
    report_every_n_ = std::max(1, static_cast<int>(this->get_parameter("report_every_n").as_int()));
    report_path_ = this->get_parameter("report_path").as_string();

    episode_event_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/" + ns + "/episode_event", 10,
      std::bind(&SuccessMonitorNode::on_episode_event, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Success monitor (C++) listening on '/%s/episode_event'.", ns.c_str());
  }

  void log_summary() const
  {
    const double rate = total_ > 0 ? (100.0 * success_ / total_) : 0.0;
    std::string failures;
    for (const auto & [reason, count] : failures_by_reason_) {
      failures += reason + "=" + std::to_string(count) + " ";
    }
    RCLCPP_INFO(
      this->get_logger(), "[gear_insertion validation] episodes=%d success=%d (%.1f%%) failures={ %s}",
      total_, success_, rate, failures.c_str());
  }

private:
  static std::string strip_slashes(const std::string & s)
  {
    size_t start = s.find_first_not_of('/');
    size_t end = s.find_last_not_of('/');
    if (start == std::string::npos) {return s;}
    return s.substr(start, end - start + 1);
  }

  void on_episode_event(const std_msgs::msg::String::SharedPtr msg)
  {
    if (!json_has_event(msg->data, "episode_end")) {
      return;
    }
    ++total_;
    auto outcome = extract_json_string_field(msg->data, "outcome");
    if (outcome.has_value() && outcome.value() == "success") {
      ++success_;
    } else {
      auto reason = extract_json_string_field(msg->data, "reason").value_or("unknown");
      ++failures_by_reason_[reason];
    }

    if (total_ % report_every_n_ == 0) {
      log_summary();
    }
    if (!report_path_.empty()) {
      write_report();
    }
  }

  void write_report() const
  {
    std::ofstream out(report_path_);
    if (!out) {
      RCLCPP_WARN(this->get_logger(), "Could not open report_path '%s' for writing.", report_path_.c_str());
      return;
    }
    out << "{\n";
    out << "  \"episodes\": " << total_ << ",\n";
    out << "  \"success\": " << success_ << ",\n";
    out << "  \"success_rate\": " << (total_ > 0 ? static_cast<double>(success_) / total_ : 0.0) << ",\n";
    out << "  \"failures_by_reason\": {";
    bool first = true;
    for (const auto & [reason, count] : failures_by_reason_) {
      if (!first) {out << ",";}
      first = false;
      out << "\n    \"" << reason << "\": " << count;
    }
    out << (failures_by_reason_.empty() ? "" : "\n  ") << "}\n";
    out << "}\n";
  }

  int total_ = 0;
  int success_ = 0;
  int report_every_n_ = 1;
  std::string report_path_;
  std::map<std::string, int> failures_by_reason_;

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr episode_event_sub_;
};

}  // namespace gear_insertion_ros2_cpp

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<gear_insertion_ros2_cpp::SuccessMonitorNode>();
  rclcpp::spin(node);
  node->log_summary();
  rclcpp::shutdown();
  return 0;
}
