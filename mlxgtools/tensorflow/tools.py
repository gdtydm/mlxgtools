import random  
import os 
import tensorflow as tf 
import numpy as np


def seed_everything_tf(seed=21):
    random.seed(seed) 
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)