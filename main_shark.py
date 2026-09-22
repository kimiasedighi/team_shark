import numpy as np
import fa2026 as fa

rng = np.random.default_rng(1)

def shark_rule(observation, memory):
    return fa.Action(
        push=rng.uniform(0.0, 0.2),
        move=tuple(rng.uniform(-0.3, 0.3, size=2)),
        pheromones=(rng.uniform(0.0, 0.01),),
    ), memory

def gradient(pher):

    values = pher[0]

    DX = np.array([[-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]])
    DY = np.array([[-1, -1, -1],       
                [ 0,  0,  0],
                [ 1,  1,  1]])
    
    dx = np.sum(DX * values)
    dy = np.sum(DY * values)

    return dx, dy

# test = np.array([[[0.9, 0.2, 1],
#                 [0.8, 0.9, 0.2],
#                 [0, 0, 0]]])
# print(test)
# print(gradient(test))