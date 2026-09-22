import numpy as np
import fa2026 as fa

rng = np.random.default_rng(1)

def shark_rule(observation, memory):
    
    return ...



def _direction_clear(vision, r, hx, hy, depth=1):
    """True if the cells up to `depth` along (hx, hy) are not walls."""
    for i in range(1, depth + 1):
        cx = int(np.round(hx * i))
        cy = int(np.round(hy * i))
        if abs(cx) > r or abs(cy) > r:
            return False
        if int(vision[r + cy, r + cx]) & int(fa.Cell.WALL):
            return False
    return True


def explore(observation, memory):
    """Persistent-heading exploration with wall bouncing."""
    vision = observation.vision
    r = vision.shape[0] // 2

    heading_x = float(memory[0])
    heading_y = float(memory[1])
    steps = float(memory[2])

    # --- initialise heading on first call ---
    if heading_x == 0.0 and heading_y == 0.0:
        angle = np.random.uniform(0.0, 2.0 * np.pi)
        heading_x = float(np.cos(angle))
        heading_y = float(np.sin(angle))
        steps = 0.0

    # --- decide whether to keep or change heading ---
    need_new = (
        steps >= 100.0
        or not _direction_clear(vision, r, heading_x, heading_y)
    )

    if need_new:
        # sample candidate directions, prefer ones that stay clear further
        best = None
        best_depth = -1
        for _ in range(12):
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            hx, hy = float(np.cos(angle)), float(np.sin(angle))
            for depth in (3, 2, 1):
                if _direction_clear(vision, r, hx, hy, depth=depth):
                    if depth > best_depth:
                        best = (hx, hy)
                        best_depth = depth
                    break
        if best is None:
            # truly boxed in: pick any direction, the wall-stop will handle it
            angle = np.random.uniform(0.0, 2.0 * np.pi)
            best = (float(np.cos(angle)), float(np.sin(angle)))
        heading_x, heading_y = best
        steps = 0.0

    # --- adaptive step size to stay within energy budget ---
    budget = max(0.0, 0.5 * observation.energy)   # use at most half of it
    step = min(0.5, float(np.sqrt(budget)))

    move = (heading_x * step, heading_y * step)

    # --- write memory ---
    new_memory = memory.copy()
    new_memory[0] = heading_x
    new_memory[1] = heading_y
    new_memory[2] = steps + 1.0
    new_memory[3] = 0.0   # mode: exploring

    return fa.Action(move=move), new_memory


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