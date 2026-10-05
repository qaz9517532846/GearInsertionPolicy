#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Isaac-Sim side of the ROS2 closed-loop validation harness.

This plays the role of "the robot" in a ROS2 AI-model validation setup, but
instead of a real UR10e/Rizon 4s on the other end of the wire, it is this
project's own ``GearInsertion-*-ROSInference-v0`` Isaac Lab environment:

    policy_inference_node (pure ROS2, ros2_validation/)
            | /gear_insertion/policy_action  (std_msgs/Float32MultiArray)
            v
    *** this script ***  (needs AppLauncher -> run with `python`, not `ros2 run`)
        - steps the Isaac Lab ManagerBasedRLEnv with the received action
        - publishes the exact same "policy" observation group used at
          training time (joint_pos, joint_vel, gear_shaft_pos, gear_shaft_quat
          -- noise included, since ObservationsCfg.PolicyCfg.enable_corruption
          is True), split back out into ROS2 messages a real perception/robot
          stack would plausibly publish
            -> /gear_insertion/joint_states    (sensor_msgs/JointState)
            -> /gear_insertion/gear_shaft_pose (geometry_msgs/PoseStamped)
        - detects episode boundaries (gear dropped / orientation exceeded /
          time-out) and publishes a JSON episode_end event so the inference
          node can reset its RNN hidden state and the success_monitor node
          can tally a pass/fail

This validates "Input Consistency" and "Output/System Response Consistency"
(see README.md) for the *exported* policy over a real ROS2 transport, using
Isaac Sim as a stand-in for the real robot + perception pipeline, before ever
touching real hardware.

Prerequisites (cannot be verified from this sandbox -- see README.md):
    - A ROS2 distro matching your Isaac Sim ROS2 bridge build (e.g. Humble)
      sourced in this shell (`source /opt/ros/<distro>/setup.bash`) *before*
      launching this script, so `import rclpy` succeeds once the
      `isaacsim.ros2.bridge` extension is enabled.

This script does NOT load a policy checkpoint itself -- that happens in the
separate ``policy_inference_node`` (see ros2_validation/), which is the whole
point: the two halves only ever talk over ROS2 topics.

Usage:
    python scripts/ros2_isaac_sim_bridge.py \\
        --task GearInsertion-UR10e-2F140-ROSInference-v0 \\
        --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The package lives under train/ (GearInsertionPolicy/train/gear_insertion_policy/).
_TRAIN_ROOT = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

# --- CLI args parsed BEFORE AppLauncher boots Kit (mirrors every other Isaac
# Lab standalone script: AppLauncher consumes its own --headless/--device/...
# args from the same parser via add_app_launcher_args). -------------------
parser = argparse.ArgumentParser(description="Isaac-Sim side of the Gear Insertion ROS2 validation loop.")
parser.add_argument(
    "--task",
    type=str,
    default="GearInsertion-UR10e-2F140-ROSInference-v0",
    help="Registered GearInsertion-*-ROSInference-v0 task to simulate as the ROS2 'robot'.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=0,
    help="Stop after this many completed episodes (0 = run forever, e.g. for interactive use).",
)
parser.add_argument(
    "--ros_namespace",
    type=str,
    default="gear_insertion",
    help="Namespace prefix for every topic this node publishes/subscribes.",
)
parser.add_argument(
    "--hold_last_action_steps",
    type=int,
    default=10,
    help=(
        "If no new /policy_action message has arrived for this many control steps, stop stepping with a stale"
        " action and instead hold the arm still (zero delta). Guards against a crashed/slow inference node"
        " silently driving the simulated robot with an ancient command."
    ),
)
parser.add_argument(
    "--insertion_distance_threshold",
    type=float,
    default=0.01,
    help=(
        "Meters. An episode that reaches time_out is only classified 'success' if the gear's final distance"
        " to its target shaft position (ground-truth, from the noise-free 'critic' observation group) is below"
        " this threshold -- otherwise it is 'failure'/'time_out_not_inserted'. Without this check, time_out"
        " alone only means 'didn't drop or over-rotate the gear', not 'actually inserted it'."
    ),
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Everything below needs the Kit app running (isaacsim.ros2.bridge, torch,
# isaaclab.envs, gymnasium, ...) -------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

# Enable the ROS2 bridge extension directly through Kit's own extension
# manager (`omni.kit.app`) rather than an isaacsim convenience wrapper: the
# wrapper's module path has already moved once between Isaac Sim releases
# (`isaacsim.core.utils.extensions.enable_extension` -> deprecated in favor of
# `isaacsim.core.experimental.utils.app.enable_extension` in newer builds).
# `omni.kit.app` is the stable underlying Kit API both wrappers call into, so
# this works unchanged across Isaac Sim versions.
import omni.kit.app  # noqa: E402

omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from rclpy.node import Node  # noqa: E402
from sensor_msgs.msg import JointState  # noqa: E402
from std_msgs.msg import Float32MultiArray, String  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import gear_insertion_policy.tasks  # noqa: E402,F401


def _obs_term_slice(env, group: str, term_name: str) -> slice:
    """Return the slice of a concatenated observation group tensor that holds one term.

    Term order/sizes vary by robot (e.g. the critic group's `joint_pos`/`joint_vel`
    cover *all* robot joints -- 14 for UR10e+2F140 (6 arm + 8 gripper), a different
    count for Rizon 4s+Grav -- not just `num_arm_joints`), so offsets are computed
    dynamically from `ObservationManager.active_terms`/`group_obs_term_dim` instead
    of being hardcoded.
    """
    names = env.observation_manager.active_terms[group]
    dims = env.observation_manager.group_obs_term_dim[group]
    start = 0
    for name, dim in zip(names, dims):
        size = int(torch.tensor(dim).prod().item())
        if name == term_name:
            return slice(start, start + size)
        start += size
    raise KeyError(f"Observation term '{term_name}' not found in group '{group}' (have: {names}).")


class IsaacSimRobotBridgeNode(Node):
    """Bridges one simulated Isaac Lab environment instance to ROS2 topics.

    Acts exactly like the ROS2-facing half of a real robot driver + perception
    stack would: it publishes sensor-like topics and consumes a commanded
    action, without knowing anything about who/what is on the other end
    (could be our ``policy_inference_node``, ``rqt``, a hand-written test
    script, or eventually a real Isaac ROS pipeline).
    """

    def __init__(self, task_id: str, ros_namespace: str, hold_last_action_steps: int, insertion_distance_threshold: float):
        super().__init__("isaac_sim_robot_bridge")

        # `gym.make(task_id, num_envs=1)` does NOT work for Isaac Lab tasks: the
        # Gym entry point (`ManagerBasedRLEnv.__init__`) takes a fully-built `cfg`
        # object, not ad-hoc kwargs. The standard pattern (mirrors every Isaac Lab
        # standalone script, e.g. scripts/environments/random_agent.py) is to load
        # and override the registered env_cfg_entry_point class first, then pass it
        # in via `cfg=`.
        env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1)
        self.env = gym.make(task_id, cfg=env_cfg).unwrapped
        self.num_arm_joints: int = self.env.cfg.num_arm_joints
        self.arm_joint_names: list[str] = list(self.env.cfg.arm_joint_names)
        self._insertion_distance_threshold = insertion_distance_threshold
        # Ground-truth (noise-free) gear/shaft positions live in the 'critic' obs
        # group -- `gear_pos` (the gear's actual world position) and
        # `gear_shaft_pos` (the target insertion point). Resolved once here via
        # `_obs_term_slice` rather than hardcoded offsets, since the critic group's
        # `joint_pos`/`joint_vel` size depends on the robot/gripper (unlike `policy`'s,
        # which is fixed to `num_arm_joints`).
        self._critic_gear_pos_slice = _obs_term_slice(self.env, "critic", "gear_pos")
        self._critic_gear_shaft_pos_slice = _obs_term_slice(self.env, "critic", "gear_shaft_pos")
        self.get_logger().info(
            f"Loaded '{task_id}' as the simulated ROS2 robot "
            f"({self.num_arm_joints} arm joints: {self.arm_joint_names}); "
            f"insertion success threshold = {insertion_distance_threshold * 1000:.1f} mm."
        )

        ns = ros_namespace.strip("/")
        self._joint_state_pub = self.create_publisher(JointState, f"/{ns}/joint_states", 10)
        self._gear_shaft_pose_pub = self.create_publisher(PoseStamped, f"/{ns}/gear_shaft_pose", 10)
        self._episode_event_pub = self.create_publisher(String, f"/{ns}/episode_event", 10)
        self._action_sub = self.create_subscription(
            Float32MultiArray, f"/{ns}/policy_action", self._on_policy_action, 10
        )

        self._latest_action: torch.Tensor | None = None
        self._steps_since_action = 0
        self._hold_last_action_steps = hold_last_action_steps
        self._episode_index = 0
        self._obs, _ = self.env.reset()
        self._publish_observations()

    def _on_policy_action(self, msg: Float32MultiArray) -> None:
        action = torch.tensor(list(msg.data), dtype=torch.float32, device=self.env.device)
        if action.numel() != self.num_arm_joints:
            self.get_logger().warn(
                f"Dropping /policy_action with {action.numel()} values, expected {self.num_arm_joints}."
            )
            return
        self._latest_action = action.unsqueeze(0)
        self._steps_since_action = 0

    def step_once(self) -> None:
        """Advance the simulation by exactly one control step (one ``env.step()`` call)."""
        if self._latest_action is not None and self._steps_since_action < self._hold_last_action_steps:
            action = self._latest_action
        else:
            # No fresh command (inference node not up yet, or stalled): hold position
            # (zero relative-joint-position delta) rather than apply a stale/unsafe action.
            action = torch.zeros((1, self.num_arm_joints), device=self.env.device)
        self._steps_since_action += 1

        self._obs, _reward, terminated, truncated, _extras = self.env.step(action)
        self._publish_observations()

        done = bool(terminated[0]) or bool(truncated[0])
        if done:
            self._publish_episode_end(terminated_timeout=bool(truncated[0]))
            self._episode_index += 1

    def _publish_observations(self) -> None:
        # `self._obs["policy"]` is the *exact* tensor the policy was trained/played
        # with (noise included, since PolicyCfg.enable_corruption=True) -- we are
        # only re-slicing it into named ROS2 fields, not recomputing it, so the
        # published topics stay bit-for-bit consistent with training-time Input
        # Consistency guarantees.
        policy_obs = self._obs["policy"][0].detach().cpu()
        n = self.num_arm_joints
        joint_pos = policy_obs[0:n]
        joint_vel = policy_obs[n : 2 * n]
        gear_shaft_pos = policy_obs[2 * n : 2 * n + 3]
        gear_shaft_quat_xyzw = policy_obs[2 * n + 3 : 2 * n + 7]

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.arm_joint_names
        js.position = joint_pos.tolist()
        js.velocity = joint_vel.tolist()
        self._joint_state_pub.publish(js)

        pose = PoseStamped()
        pose.header.stamp = js.header.stamp
        pose.header.frame_id = "robot_base"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = gear_shaft_pos.tolist()
        # NOTE on quaternion order: `gear_shaft_quat_w()` (mdp/observations.py) reads
        # `RigidObjectData.root_quat_w`, whose element order is defined by
        # `QUAT_XYZW_ELEMENT_NAMES = ["qx", "qy", "qz", "qw"]` in
        # isaaclab/utils/leapp/leapp_semantics.py -- i.e. Isaac Lab 3.0's `ProxyArray`
        # quaternions are (x, y, z, w), the same order as geometry_msgs/Quaternion.
        # (This is a real behavior change vs. older Isaac Lab releases that used
        # (w, x, y, z); the docstring on gear_shaft_quat_w() in this copy is stale on
        # that point -- the `base_quat[:, 3] < 0` sign-canonicalization check inside
        # it is only correct if index 3 is w, which confirms xyzw at runtime.) So no
        # component reordering is needed here -- do NOT blindly assume (w,x,y,z) when
        # wiring a different observation term or a real perception node.
        x, y, z, w = gear_shaft_quat_xyzw.tolist()
        pose.pose.orientation.x = x
        pose.pose.orientation.y = y
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        self._gear_shaft_pose_pub.publish(pose)

    def _publish_episode_end(self, terminated_timeout: bool) -> None:
        term_mgr = self.env.termination_manager
        dropped = bool(term_mgr.get_term("gear_dropped")[0])
        mis_oriented = bool(term_mgr.get_term("gear_orientation_exceeded")[0])

        # Ground-truth distance between the gear's actual position and its target
        # shaft position, from the noise-free 'critic' observation group (NOT the
        # noisy 'policy' group the policy itself sees). Still meaningful to compute
        # even on a dropped/mis-oriented episode (useful diagnostic), but only used
        # to gate the outcome in the time_out branch below.
        critic_obs = self._obs["critic"][0].detach().cpu()
        gear_pos = critic_obs[self._critic_gear_pos_slice]
        gear_shaft_pos = critic_obs[self._critic_gear_shaft_pos_slice]
        final_distance_m = torch.linalg.norm(gear_pos - gear_shaft_pos).item()

        if dropped:
            outcome, reason = "failure", "gear_dropped"
        elif mis_oriented:
            outcome, reason = "failure", "gear_orientation_exceeded"
        elif terminated_timeout:
            # Strict check: time_out alone only means "didn't drop or over-rotate
            # the gear" -- it does NOT verify the gear actually reached the shaft.
            # Gate on the ground-truth gear-to-shaft distance too.
            if final_distance_m < self._insertion_distance_threshold:
                outcome, reason = "success", "time_out"
            else:
                outcome, reason = "failure", "time_out_not_inserted"
        else:
            outcome, reason = "failure", "unknown"

        payload = {
            "event": "episode_end",
            "episode_index": self._episode_index,
            "outcome": outcome,
            "reason": reason,
            "final_distance_m": round(final_distance_m, 5),
        }
        msg = String()
        msg.data = json.dumps(payload)
        self._episode_event_pub.publish(msg)
        self.get_logger().info(
            f"Episode {self._episode_index}: {outcome} ({reason}), final_distance={final_distance_m * 1000:.1f}mm"
        )


def main() -> None:
    rclpy.init(args=[])
    bridge = IsaacSimRobotBridgeNode(
        args_cli.task, args_cli.ros_namespace, args_cli.hold_last_action_steps, args_cli.insertion_distance_threshold
    )
    try:
        while simulation_app.is_running():
            rclpy.spin_once(bridge, timeout_sec=0.0)
            bridge.step_once()
            if args_cli.num_episodes and bridge._episode_index >= args_cli.num_episodes:
                break
    finally:
        bridge.destroy_node()
        rclpy.shutdown()
        simulation_app.close()


if __name__ == "__main__":
    main()
