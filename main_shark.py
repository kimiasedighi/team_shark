import numpy as np
import fa2026 as fa

rng = np.random.default_rng(1)

def shark_rule(observation, memory):
    return fa.Action(
        push=rng.uniform(0.0, 0.2),
        move=tuple(rng.uniform(-0.3, 0.3, size=2)),
        pheromones=(rng.uniform(0.0, 0.01),),
    ), memory