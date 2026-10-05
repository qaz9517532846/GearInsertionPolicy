// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <optional>
#include <string>

namespace gear_insertion_ros2_cpp {

// Minimal, dependency-free extraction of `"field": "value"` / `"field": <number>`
// out of the *fixed-format* JSON this project's own producer emits
// (see scripts/ros2_isaac_sim_bridge.py::_publish_episode_end, which always
// writes via Python's `json.dumps`, i.e. `{"key": "value", "key2": 1, ...}`
// with no nested objects/arrays). This is intentionally NOT a general JSON
// parser -- if you change the episode_event schema to something nested, pull
// in a real JSON library (e.g. nlohmann::json) instead of extending this.
inline std::optional<std::string> extract_json_string_field(const std::string & data, const std::string & field)
{
  const std::string key = "\"" + field + "\":";
  auto key_pos = data.find(key);
  if (key_pos == std::string::npos) {
    return std::nullopt;
  }
  auto value_start = data.find('"', key_pos + key.size());
  if (value_start == std::string::npos) {
    return std::nullopt;
  }
  ++value_start;
  auto value_end = data.find('"', value_start);
  if (value_end == std::string::npos) {
    return std::nullopt;
  }
  return data.substr(value_start, value_end - value_start);
}

inline bool json_has_event(const std::string & data, const std::string & event_name)
{
  auto event = extract_json_string_field(data, "event");
  return event.has_value() && event.value() == event_name;
}

}  // namespace gear_insertion_ros2_cpp
