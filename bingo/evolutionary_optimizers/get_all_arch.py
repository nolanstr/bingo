import dill
import numpy as np
import glob

def get_all_archipelagos(filename):
    with open(filename, "rb") as f:
        all_archipelagos = dill.load(f)
    return all_archipelagos

def get_all_islands(filename):
    all_archipelagos = get_all_archipelagos(filename)
    return [arch.island for arch in all_archipelagos]

def get_oldest_arch(DIR):
    files = glob.glob(DIR+'/*.pkl')
    archs = [get_all_archipelagos(filename) for filename in files]
    ages = [arch[0].generational_age for arch in archs]
    oldest_idx = np.argmax(ages)
    oldest_arch = archs[oldest_idx]
    return oldest_arch
    
def get_oldest_hof(DIR):
    oldest_arch =get_oldest_arch(DIR) 
    return oldest_arch[0].hall_of_fame

if __name__ == "__main__":
    pkl_path = "my_path/my_pkl.pkl"
    all_islands = get_all_islands(pkl_path)

