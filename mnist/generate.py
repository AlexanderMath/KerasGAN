import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import sys

"""
	Takes a link to a model file and uses it to generate images. 

	Call examples: 

		python generate.py models/G_vanilla_all.h5
		python generate.py models/G_vanilla_5.h5
		python generate.py models/G_ls_all.h5
		python generate.py models/G_ls_5.h5
		python generate.py models/G_wasserstein_all.h5
		python generate.py models/G_wasserstein_5.h5

"""

if len(sys.argv) == 1: 
	print("-"*80)
	print("Call exampes: ")
	print("-"*80)
	print("python generate.py models/G_vanilla_all.h5")
	print("python generate.py models/G_vanilla_5.h5")
	print("python generate.py models/G_ls_all.h5")
	print("python generate.py models/G_ls_5.h5")
	print("python generate.py models/G_wasserstein_all.h5")
	print("python generate.py models/G_wasserstein_5.h5")
	sys.exit(0)

rand_dim 	= 64
path 		= sys.argv[1]
G 			= load_model(path)
rows, cols 	= 4, 4
fig, ax 	= plt.subplots(rows, cols)
first 		= True

while True: 

	Z 		= np.random.uniform(-1, +1, size=(rows * cols, rand_dim))
	fakes 	= G.predict(Z)

	for i in range(rows): 
		for j in range(cols): 
			ax[i, j].imshow(fakes[i*rows+j].reshape(28, 28), cmap="gray")

			ax[i,j].set_xticks([])
			ax[i,j].set_yticks([])
			ax[i,j].set_xticklabels([])
			ax[i,j].set_yticklabels([])

	if first: 
		plt.tight_layout()
		first = False

	plt.pause(.1)
	input("Press any key to produce new batch of fake images. ")

