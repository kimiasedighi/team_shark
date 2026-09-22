import numpy as np
import fa2026 as fa

from memory import set_memory, get_memory

rng = np.random.default_rng(1)



def shark_rule(observation, memory):


    found_target = get_memory(memory)
    sees_target, target_direction = see_target(observation)

    #if not target and not pheromones --> explore
    #if not target but pheromones --> set weaker pheromone, go to target


    #if see target --> set pheromone, go to target
    if sees_target and not found_target:
        found_target = True
        dx, dy = target_direction
        n = float(np.hypot(dx, dy))
        move = (0.0, 0.0) if n < 1e-9 else (0.3 * dx / n, 0.3 * dy / n)
        pheromones=(0.01,)

    memory = set_memory(memory, found_target)
    return fa.Action(
        push=0.0,
        move=move,
        pheromones=pheromones,
    ), memory


def see_target(observation):
    """Return (sees, (dx, dy)): True + mean cell-offset to visible targets, else (False, (0, 0))."""
    mask = (observation.vision & int(fa.Cell.TARGET)) != 0
    if not np.any(mask):
        return False, (0.0, 0.0)
    r = observation.vision.shape[0] // 2
    ys, xs = np.nonzero(mask)
    return True, (float(xs.mean() - r), float(ys.mean() - r))
