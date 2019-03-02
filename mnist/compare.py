import numpy as np
import matplotlib.pyplot as plt
import imageio
from natsort import natsorted
import os 

clss = "all"

vanilla = np.load("images/dcgan_vanilla_loss/fake_history_%s.npz"%clss)['arr_0']
ls 		= np.load("images/dcgan_ls_loss/fake_history_%s.npz"%clss)['arr_0']
wass	= np.load("images/dcgan_wasserstein_loss/fake_history_%s.npz"%clss)['arr_0']

plt.set_cmap("gray")

fig, ax = plt.subplots(4, 3, figsize=(4.5, 6))

steps = 20
images = []

for time in range(0, len(vanilla), steps): 
	
	for i in range(4): 
		ax[i, 0].imshow(vanilla[time][i].reshape(28, 28))
		ax[i, 1].imshow(ls[time][i].reshape(28, 28))
		ax[i, 2].imshow(wass[time][i].reshape(28, 28))

		for j in range(3): 
			ax[i,j].set_xticks([])
			ax[i,j].set_yticks([])
			ax[i,j].set_xticklabels([])
			ax[i,j].set_yticklabels([])

	ax[0, 0].set_title("Vanilla")
	ax[0, 1].set_title("Least Squares")
	ax[0, 2].set_title("Wasserstein")

	fig.suptitle("[%i / %i]"%(time, len(vanilla)))

	if time == 0: plt.tight_layout(rect=[0, 0, 0, 0.1])

	plt.savefig("/tmp/gan_%i.jpg"%time)
	plt.pause(.1)
	images.append(imageio.imread("/tmp/gan_%i.jpg"%time))


imageio.mimsave("gif.gif", images, duration=.1)
