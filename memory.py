import numpy as np

def get_memory(memory : np.ndarray):
    # first float
    found_target = bool(memory[0])
    
    return found_target

def set_memory(memory : np.ndarray, found_target : bool):
    #first float
    memory[0] = float(found_target)
    return