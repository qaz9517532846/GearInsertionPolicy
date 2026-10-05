#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Plain-``python`` playback/evaluation entry point for the Gear Insertion Policy project.

Loads a trained checkpoint and rolls it out in the Isaac Sim viewer (or
headless + ``--video``) so you can visually verify the policy before
attempting real-robot deployment. Forwards to
``isaaclab_rl.entrypoints.run_play_cli`` after registering this project's own
Gym task IDs, exactly like ``scripts/train.py`` does for training.

Example:

    python GearInsertionPolicy/scripts/play.py \\
        --rl_library rsl_rl \\
        --task GearInsertion-UR10e-2F140-v0 \\
        --num_envs 16 \\
        --checkpoint logs/rsl_rl/gear_assembly_ur10e/<run>/model_<iter>.pt

``--checkpoint`` also accepts the selectors documented by IsaacLab, e.g.
``latest`` / ``best`` in place of an explicit path (see the console help of
``--rl_library rsl_rl -h`` for the current list).

Playback also writes ``policy.pt`` (TorchScript / JIT) and ``policy.onnx``
next to the checkpoint, ready to hand to the Isaac ROS inference node.
"""

from __future__ import annotations

import sys
from pathlib import Path

# The package lives under train/ (GearInsertionPolicy/train/gear_insertion_policy/).
_TRAIN_ROOT = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

import gear_insertion_policy.tasks  # noqa: E402,F401

from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning playback backend (default: rsl_rl)."""
    return run_play_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
