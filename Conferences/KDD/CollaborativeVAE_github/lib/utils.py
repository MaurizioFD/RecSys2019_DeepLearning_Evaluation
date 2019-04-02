import numpy as np
import logging

def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)

def get_batch(X, size):
    ids = np.random.choice(len(X), size, replace=False)
    return (X[ids], ids)
        
def noise_validator(noise, allowed_noises):
    '''Validates the noise provided'''
    try:
        if noise in allowed_noises:
            return True
        elif noise.split('-')[0] == 'mask' and float(noise.split('-')[1]):
            t = float(noise.split('-')[1])
            if t >= 0.0 and t <= 1.0:
                return True
            else:
                return False
    except:
        return False
    pass
