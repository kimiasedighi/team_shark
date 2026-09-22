from __future__ import annotations

import numpy as np

import fa2026 as fa

RADIUS = 4
CENTER = RADIUS  # index of the agent's own cell inside the 9x9 vision patch
WALL_BIT = int(fa.Cell.WALL)
TARGET_BIT = int(fa.Cell.TARGET)


def following_wall_rule(seed: int = 1, speed: float = 0.7, inward_bias: float = 0.15):
    """Wall-following search, with direct homing once the target is visible.

    Relies on the target always sitting on the arena's boundary wall: an
    agent that keeps a wall on one side and follows it is guaranteed to
    eventually pass right by the target. See ``getting_started.ipynb``
    section 7 for the full walkthrough.
    """
    rng = np.random.default_rng(seed)

    def rule(observation, memory):
        memory = np.array(memory, dtype=float)
        vision = observation.vision

        if vision[CENTER, CENTER] & TARGET_BIT:
            # Already inside the target: stop and wait for the rest of the swarm.
            return fa.Action(move=(0.0, 0.0)), memory

        ys, xs = np.nonzero(vision & TARGET_BIT)
        if ys.size > 0:
            # Target visible nearby: head straight for the closest target cell.
            dy, dx = ys.astype(float) - CENTER, xs.astype(float) - CENTER
            i = int(np.argmin(dx * dx + dy * dy))
            tx, ty = dx[i], dy[i]
            length = max(float(np.hypot(tx, ty)), 1e-6)
            step = min(speed, length)
            return fa.Action(move=(tx / length * step, ty / length * step)), memory

        heading_x, heading_y = memory[0], memory[1]

        # Wall contact in the immediate 3x3 neighborhood: follow it. Rotating
        # the outward wall normal by a fixed 90 degrees gives every agent the
        # same sense of travel, so the swarm circles the arena the same way
        # instead of colliding head-on along the boundary.
        near = vision[CENTER - 1 : CENTER + 2, CENTER - 1 : CENTER + 2]
        nx, ny = 0.0, 0.0
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                if (ox or oy) and near[oy + 1, ox + 1] & WALL_BIT:
                    nx, ny = nx + ox, ny + oy

        if nx or ny:
            nlen = float(np.hypot(nx, ny))
            nx, ny = nx / nlen, ny / nlen
            heading_x, heading_y = -ny, nx
            move = (
                heading_x * speed - nx * inward_bias,
                heading_y * speed - ny * inward_bias,
            )
        else:
            if heading_x == 0.0 and heading_y == 0.0:
                angle = rng.uniform(0.0, 2.0 * np.pi)
                heading_x, heading_y = float(np.cos(angle)), float(np.sin(angle))
            move = (heading_x * speed, heading_y * speed)

        memory[0], memory[1] = heading_x, heading_y
        return fa.Action(move=move), memory

    return rule


# Add make_rule_two, make_rule_three, ... here as you try other strategies,
# and register them below so the notebook can list/select them by name.
RULES = {
    "rule_one": following_wall_rule,
}
