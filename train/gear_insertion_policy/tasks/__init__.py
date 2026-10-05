"""Task registration package.

Importing this package registers every Gym environment ID defined below by
importing the per-robot ``config`` sub-packages, each of which calls
``gymnasium.register(...)``. Nothing else in this project triggers
registration automatically — ``scripts/train.py`` and ``scripts/play.py``
both do ``import gear_insertion_policy.tasks`` before touching Gym/RSL-RL so
that the IDs below exist by the time a backend calls ``gym.make(task_id)``.

Registered environment IDs
---------------------------
UR10e + Robotiq 2F-140 gripper:
    - ``GearInsertion-UR10e-2F140-v0``
    - ``GearInsertion-UR10e-2F140-ROSInference-v0`` (fixed/deterministic reset
      poses + the extra metadata Isaac ROS needs for on-robot inference)

UR10e + Robotiq 2F-85 gripper:
    - ``GearInsertion-UR10e-2F85-v0``
    - ``GearInsertion-UR10e-2F85-ROSInference-v0``

Flexiv Rizon 4s + Grav gripper:
    - ``GearInsertion-Rizon4s-Grav-v0``
    - ``GearInsertion-Rizon4s-Grav-ROSInference-v0``

To add a new robot/gripper combination, add a sibling package under
``config/`` following the same pattern (see README.md, "Swapping the Robot
Arm").
"""

from .config import rizon_4s, ur_10e  # noqa: F401
