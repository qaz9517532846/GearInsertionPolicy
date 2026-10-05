"""Gym registration for the Flexiv Rizon 4s + Grav gripper gear insertion environments."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

# Flexiv Rizon 4s + Grav gripper
gym.register(
    id="GearInsertion-Rizon4s-Grav-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:Rizon4sGearAssemblyEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGearAssemblyRNNPPORunnerCfg",
    },
)

# Flexiv Rizon 4s + Grav gripper - ROS inference (fixed poses + ROS metadata)
gym.register(
    id="GearInsertion-Rizon4s-Grav-ROSInference-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ros_inference_env_cfg:Rizon4sGearAssemblyROSInferenceEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Rizon4sGearAssemblyRNNPPORunnerCfg",
    },
)
