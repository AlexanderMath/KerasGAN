import sys
sys.path.insert(0, '../power_iteration/')
# for fast computation of largest eigenvalue
from power_iteration import power_iteration 
import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Activation, LeakyReLU, ReLU
from keras.layers import Dense, Input, Conv2D, Reshape, Flatten, BatchNormalization, Layer, Deconvolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
import tensorflow as tf

"""
	Takes arguments

		- loss function from ['vanilla', 'ls', 'wasserstein']
		- clss from 0-9, any string is interpreted as 'all' 
		  which trains on all classes. 

	It then trains a GAN on the specified number of classes on MNIST. 
	The result is visualized during training with matplotlib. 
	A few fake images a saved at each iteration. 

	Call examples: 

		python dcgan.py vanilla 4
		python dcgan.py vanilla all
		python dcgan.py ls 8
		python dcgan.py ls all
		python dcgan.py wasserstein 9
		python dcgan.py wasserstein all
		python dcgan.py all 3
		python dcgan.py all all

"""
def train(size = 32): 

	plt.ion()

	fig, ax_fakes = plt.subplots(4, 3, figsize=(5, 8))
	fig.canvas.manager.window.wm_geometry("+0+0")

	def plot(titleA="", titleB=""): 

		for k in range(4): 
			for j in range(3): 
				ax_fakes[k, j].imshow(fakes[k*4+j].reshape(img_shape))
				ax_fakes[k, j].axis('off')

		plt.title("[%i / %i]"%(i, iterations))
		fig.savefig("images/dcgan_%i/%i.jpg"%(size, i))
		plt.pause(.1)

	# Load MNIST data. 
	X_train 	= np.load("data/guitar_%i.npz"%size)['arr_0']
	img_shape 	= X_train[0].shape
	print(X_train.shape)
		
	# Normalize to [-1, +1] 
	X_train = X_train / 127.5 - 1
	assert -1.0 <= X_train.min() and X_train.max() <= 1.0, (X_train.min(), X_train.max())

	# Training parameters. 
	batch_size 	= 128
	rand_dim	= 64

	if size == 32: 		
		step_size 	= 0.001
		iterations 	= 4000

	elif size == 64: 
		step_size = 0.001 
		iterations = 10000

	# Define generator architecture. 
	G = Sequential()
	G.add(Dense(128 * 8 * 8, input_dim=(rand_dim )))
	G.add(Reshape((8, 8, 128)))
	G.add(ReLU())
	G.add(BatchNormalization())

	# size: 8 x 8 
	G.add(Deconvolution2D(64, (3, 3), strides=(2, 2), padding="same"))
	G.add(ReLU())
	G.add(BatchNormalization())
	# size: 16 x 16


	if size == 32: 
		# size: 16 x 16
		G.add(Deconvolution2D(3, (3, 3), strides=(2, 2), padding="same"))
		G.add(Reshape(img_shape))
		G.add(Activation('tanh')) # no batchnorm here as adviced in dcgan article. 
		# size: 32 x 32
	elif size == 64: 
		# size: 16 x 16
		G.add(Deconvolution2D(32, (3, 3), strides=(2, 2), padding="same"))
		G.add(ReLU())
		G.add(BatchNormalization())
		# size: 32 x 32

		# size: 32 x 32 
		G.add(Deconvolution2D(3, (3, 3), strides=(2, 2), padding="same"))
		G.add(Reshape(img_shape))
		G.add(Activation('tanh')) # no batchnorm here as adviced in dcgan article. 
		# size: 64 x 64


	# Get summary. 
	G.predict(np.random.normal(0, 1, size=(10, 64)))
	G.summary()

	# Define discriminator architecture: 
	D = Sequential()

	if size == 32: 
		# size: 32 x 32
		D.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
		D.add(LeakyReLU(0.2))
		D.add(BatchNormalization())
		# size: 16 x 16
	elif size == 64: 
		# size: 64 x 64
		D.add(Conv2D(32, (3, 3), strides=(2, 2), padding="same"))
		D.add(LeakyReLU(0.2))
		D.add(BatchNormalization())
		# size: 32 x 32

		# size: 32 x 32
		D.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
		D.add(LeakyReLU(0.2))
		D.add(BatchNormalization())
		# size: 16 x 16

	# size: 16 x 16
	D.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
	D.add(LeakyReLU(0.2))
	D.add(BatchNormalization())
	# size: 8 x 8

	D.add(Flatten())
	D.add(Dense(1))

	# Get summary. 
	D.predict(X_train[:2])
	D.summary()

	# Define the different loss for discriminator. 
	optimizer = Adam(step_size, beta_1=0.5)
	D.compile(loss="mae", 		optimizer=optimizer)

	# Freeze discriminator weights so we can use it as *fixed* loss function
	# for the generator. 
	D.trainable = False

	# Define a new model that connects the generator with the discriminator 
	# as a loss function. 
	train_G = Sequential()
	train_G.add(G)
	train_G.add(D)

	train_G.compile(loss="mae", optimizer=optimizer)

	yes_labels 	= np.ones(batch_size)
	no_labels 	= np.zeros(batch_size)

	# Save a list of fakes and save in end. 
	fake_history = [] 

	# Init vectors for power iteration should initially be None 
	v1, v2, v3, v4 = None, None, None, None

	for i in range(iterations):
		print("\r--- [%i / %i] ---"%(i, iterations), end="")
		
		X = X_train[np.random.permutation(X_train.shape[0])[:batch_size]]#np.random.normal(-4, 1, size=(n, 1))
		Z = np.random.uniform(-1, +1, size=(batch_size, rand_dim))

		fakes = G.predict(Z)

		D.train_on_batch(X, 	yes_labels)
		D.train_on_batch(fakes, no_labels)

		# Spectral normalization: 
		# It is not obvious how we should treat convolutions as matrices, for now I 
		# use the approach outlined in the original article.  Maybe if we analyze it using 
		# the convolution theorem we can do it more fine grained and show that this works better? 
		if size == 64: 
			W1 = D.get_weights()[0]  # 3 x 3 x 3 x 32
			W2 = D.get_weights()[6]  # 3 x 3 x 32 x 64
			W3 = D.get_weights()[12] # 3 x 3 x 64 x 128 
			W4 = D.get_weights()[18] # FC ; vector product

			W1 = W1.reshape(32, 	3*3*3) 	# d_out = 64, 	w=h=3, d_in = 3
			W2 = W2.reshape(64, 	3*3*32) # d_out = 128, 	w=h=3, d_in = 64
			W3 = W3.reshape(128, 	3*3*64) # d_out = 128, 	w=h=3, d_in = 64

			# slow naive method, computes SVD each iteration. 
			#W1 = W1 / np.linalg.svd(W1, full_matrices=False, compute_uv=False).max()
			#W2 = W2 / np.linalg.svd(W2, full_matrices=False, compute_uv=False).max()
			#W3 = W3 / np.linalg.svd(W3, full_matrices=False, compute_uv=False).max()

			# computes just largest eigenvalue and reuses largest eigenvector as initialization. 
			s1, v1 = power_iteration(W1, num_simulations=3, init_vect=v1, return_vector=True)
			s2, v2 = power_iteration(W2, num_simulations=3, init_vect=v2, return_vector=True)
			s3, v3 = power_iteration(W3, num_simulations=3, init_vect=v3, return_vector=True)
			s4, v4 = power_iteration(W4, num_simulations=3, init_vect=v4, return_vector=True)
			W1 = W1 / s1
			W2 = W2 / s2
			W3 = W3 / s3
			W4 = W4 / s4

			weights = D.get_weights()
			weights[0] = W1.reshape(3, 3, 3, 	32)
			weights[6] = W2.reshape(3, 3, 32, 	64)
			weights[12] = W3.reshape(3, 3, 64, 	128)
		if size == 32: 
			W1 = D.get_weights()[0]  # 3 x 3 x 3 x 64
			W2 = D.get_weights()[6] # 3 x 3 x 64 x 128 
			W3 = D.get_weights()[12] # FC ; vector product

			W1 = W1.reshape(64, 	3*3*3) # d_out = 128, 	w=h=3, d_in = 64
			W2 = W2.reshape(128, 	3*3*64) # d_out = 128, 	w=h=3, d_in = 64

			# slow naive method, computes SVD each iteration. 
			#W1 = W1 / np.linalg.svd(W1, full_matrices=False, compute_uv=False).max()
			#W2 = W2 / np.linalg.svd(W2, full_matrices=False, compute_uv=False).max()
			#W3 = W3 / np.linalg.svd(W3, full_matrices=False, compute_uv=False).max()

			# computes just largest eigenvalue and reuses largest eigenvector as initialization. 
			s1, v1 = power_iteration(W1, num_simulations=3, init_vect=v1, return_vector=True)
			s2, v2 = power_iteration(W2, num_simulations=3, init_vect=v2, return_vector=True)
			s3, v3 = power_iteration(W3, num_simulations=3, init_vect=v3, return_vector=True)
			W1 = W1 / s1
			W2 = W2 / s2
			W3 = W3 / s3

			weights = D.get_weights()
			weights[0] = W1.reshape(3, 3, 3, 	64)
			weights[6] = W2.reshape(3, 3, 64, 	128)



		train_G.train_on_batch(Z, 	yes_labels)

		if i % 200 == 0: plot()

		if size == 32: 
			if i % 10 == 0: fake_history.append(fakes[:4]) 

		if size == 64: 
			if i % 100 == 0: fake_history.append(fakes[:4]) # this can take up too much ram. 

	D.save("models/gan_D_%i.h5"%size)
	G.save("models/gan_G_%i.h5"%size)
	np.savez("images/dcgan_%i/fake_history"%size, fake_history)


if __name__ == "__main__":
	def help(): 
		print("-"*80)
		print("Examples of calling 'dcgan.py'. ")
		print("-"*80)
		print("dcgan.py 32")
		print("dcgan.py 64")

	if len(sys.argv) == 1: 	size = 32
	else: 					size = int(sys.argv[1])

	train(size)
