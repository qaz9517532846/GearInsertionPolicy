"""Gym registration for the UR10e gear insertion environments.

Two gripper variants (Robotiq 2F-140 / 2F-85) x two flavors each:
    - the plain training variant (randomized resets, used for RL training)
    - the "ROSInference" variant (fixed/deterministic resets + extra metadata
      consumed by the Isaac ROS inference node on the real robot)
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# UR10e with Robotiq 2F-140 gripper
gym.register(
    id="GearInsertion-UR10e-2F140-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F140GearAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)

# UR10e with Robotiq 2F-85 gripper
gym.register(
    id="GearInsertion-UR10e-2F85-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10e2F85GearAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)

# UR10e with Robotiq 2F-140 gripper - ROS inference (fixed poses + ROS metadata)
gym.register(
    id="GearInsertion-UR10e-2F140-ROSInference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:UR10e2F140GearAssemblyROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)

# UR10e with Robotiq 2F-85 gripper - ROS inference (fixed poses + ROS metadata)
gym.register(
    id="GearInsertion-UR10e-2F85-ROSInference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:UR10e2F85GearAssemblyROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10GearAssemblyRNNPPORunnerCfg",
    },
)
