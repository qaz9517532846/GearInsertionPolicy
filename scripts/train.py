#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Plain-``python`` training entry point for the Gear Insertion Policy project.

This replaces the ``isaaclab.sh train`` / ``uv run isaaclab train`` wrappers used
by upstream Isaac Lab. It does the exact same thing they do (it forwards to
``isaaclab_rl.entrypoints.run_train_cli``, the same unified RL entrypoint used by
``scripts/reinforcement_learning/train.py`` in the IsaacLab repo), with one
addition: it imports this project's own ``gear_insertion_policy.tasks`` package
*first*, so the ``GearInsertion-*`` Gym IDs are registered before RSL-RL ever
calls ``gym.make(task_id)``.

Must be run with the Python interpreter that has Isaac Lab installed (the
IsaacLab venv / conda env / ``isaaclab.sh -p``), e.g.:

    python GearInsertionPolicy/scripts/train.py \\
        --rl_library rsl_rl \\
        --task GearInsertion-UR10e-2F140-v0 \\
        --num_envs 256 --headless

See ../README.md for the full list of registered tasks and the recommended
training SOP (small-scale visual check -> full headless run -> TensorBoard).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the local `gear_insertion_policy` package importable without installing it
# (so this script works straight out of a git checkout/copy of this folder).
# The package lives under train/ (GearInsertionPolicy/train/gear_insertion_policy/).
_TRAIN_ROOT = Path(__file__).resolve().parent.parent / "train"
if str(_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_TRAIN_ROOT))

# Warp captures `enable_backward` at import time (mirrors upstream train.py);
# Isaac Lab does not use Warp autodiff, so disabling it halves kernel build time
# on a cold cache. Must happen before anything that defines Warp kernels is imported.
import warp as wp  # noqa: E402

wp.config.enable_backward = False

# Registers GearInsertion-UR10e-*/GearInsertion-Rizon4s-* with gymnasium.
import gear_insertion_policy.tasks  # noqa: E402,F401

from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the selected reinforcement learning training backend (default: rsl_rl)."""
    return run_train_cli(argv)


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
