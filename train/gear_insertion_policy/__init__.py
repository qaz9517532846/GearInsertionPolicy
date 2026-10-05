"""Gear Insertion Policy — a standalone, editable copy of Isaac Lab's gear assembly
sim-to-real task (see https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html).

This package intentionally lives *outside* ``isaaclab_tasks`` so it can be trained,
read, and modified with plain ``python`` commands, without editing the IsaacLab
installation itself. It still imports IsaacLab's core libraries (``isaaclab``,
``isaaclab_assets``, ``isaaclab_tasks``, ``isaaclab_rl``, ``isaaclab_physx``) from
whatever IsaacLab environment/venv is active, so it must be run with the same
Python interpreter that has IsaacLab installed.

Sub-packages:
    ``tasks``: the Gym-registered Manager-Based RL environment definitions
    (scene, observations, rewards, terminations, events) and the two supported
    robot configurations (UR10e, Flexiv Rizon 4s).

See ``README.md`` at the project root for the full setup/training/deployment SOP
and a guide for swapping in a different robot arm or manipulated object.
"""

__version__ = "0.1.0"
