import numpy as np
import skvideo.io
import matplotlib.pyplot as plt
import time
import sys

size = int(sys.argv[1])

t0 = time.time()
data = skvideo.io.vread("output_%i.avi"%size)
def rgb2gray(rgb): return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
print(data.shape)
data_gray = rgb2gray(data).reshape(data.shape[0], size, size, 1)
data_gray = np.concatenate((data_gray, data_gray, data_gray), axis=-1)
print(data_gray.shape)
print("Loaded video in time: \t%.4fs"%(time.time() - t0))
np.savez_compressed("guitar_%i.npz"%size, data, data_gray)
