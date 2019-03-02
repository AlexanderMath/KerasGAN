import os
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Activation, LeakyReLU, ReLU
from keras.layers import Dense, Input, Conv2D, Reshape, Flatten, BatchNormalization, Layer, Deconvolution2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt 
import tensorflow as tf
import sys

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
def train(loss, clss): 

	plt.set_cmap("gray")
	plt.ion()

	fig, ax_fakes = plt.subplots(4, 3, figsize=(5, 8))
	fig.canvas.manager.window.wm_geometry("+0+0")

	def plot(titleA="", titleB=""): 

		for k in range(4): 
			for j in range(3): 
				ax_fakes[k, j].imshow(fakes[k*4+j].reshape(28, 28))
				ax_fakes[k, j].axis('off')

		plt.title("[%i / %i]"%(i, iterations))
		fig.savefig("images/dcgan_%s_loss/class_%s_num_%i.jpg"%(loss, clss, i))
		plt.pause(.1)

	# Load MNIST data. 
	from keras.datasets import mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
	print(x_train.shape)
		
	# Normalize to [-1, +1] 
	x_train = x_train / 127.5 - 1
	assert -1.0 <= x_train.min() and x_train.max() <= 1.0, (x_train.min(), x_train.max())

	# Choose a single class or all classes. 
	if clss.isnumeric(): 	X_train = x_train[y_train == int(clss)] # any string and we try to learn on all. 
	else:					X_train = x_train

	# Training parameters. 
	batch_size 	= 128
	iterations 	= 2000
	rand_dim	= 64

	# Define generator architecture. 
	G = Sequential()
	G.add(Dense(128 * 7 * 7, input_dim=(rand_dim )))
	G.add(Reshape((7, 7, 128)))
	G.add(ReLU())
	G.add(BatchNormalization())

	# size: 7 x 7 
	G.add(Deconvolution2D(64, (3, 3), strides=(2, 2), padding="same"))
	G.add(ReLU())
	G.add(BatchNormalization())
	# size: 14 x 14

	# size: 14 x 14
	G.add(Deconvolution2D(1, (3, 3), strides=(2, 2), padding="same"))
	G.add(Reshape((28, 28, 1)))
	G.add(Activation('tanh')) # no batchnorm here as adviced in dcgan article. 
	# size: 28 x 28

	# Get summary. 
	G.predict(np.random.normal(0, 1, size=(10, 64)))
	G.summary()


	# Define discriminator architecture: 
	D = Sequential()
	# size: 28 x 28
	D.add(Conv2D(64, (3, 3), strides=(2, 2), padding="same"))
	D.add(LeakyReLU(0.2))
	D.add(BatchNormalization())
	# size: 14 x 14

	# size: 14 x 14
	D.add(Conv2D(128, (3, 3), strides=(2, 2), padding="same"))
	D.add(LeakyReLU(0.2))
	D.add(BatchNormalization())
	# size: 7 x 7

	D.add(Flatten())
	D.add(Dense(1))

	# The vanilla GAN uses 'sigmoid' to get a number between [0,1] 
	# which is interpreted as an probability. Both least squares 
	# and wasserstein GAN does not use this. 
	if loss == "vanilla": D.add(Activation("sigmoid"))

	# Get summary. 
	D.predict(x_train[:2])
	D.summary()

	# Define the different loss for discriminator. 
	optimizer = Adam(0.0002, beta_1=0.5)
	if loss == "ls": 			D.compile(loss="mse", 					optimizer=optimizer)
	if loss == "vanilla": 		D.compile(loss="binary_crossentropy", 	optimizer=optimizer)
	if loss == "wasserstein": 	D.compile(loss="mae", 					optimizer=optimizer)

	# Freeze discriminator weights so we can use it as *fixed* loss function
	# for the generator. 
	D.trainable = False

	# Define a new model that connects the generator with the discriminator 
	# as a loss function. 
	train_G = Sequential()
	train_G.add(G)
	train_G.add(D)

	if loss == "ls": 			train_G.compile(loss="mse", 				optimizer=optimizer)
	if loss == "vanilla": 		train_G.compile(loss="binary_crossentropy", optimizer=optimizer)
	if loss == "wasserstein": 	train_G.compile(loss="mae", 				optimizer=optimizer)

	yes_labels 	= np.ones(batch_size)
	no_labels 	= np.zeros(batch_size)

	# Save a list of fakes and save in end. 
	fake_history = [] 

	for i in range(iterations):
		print("\r--- [%i / %i] ---"%(i, iterations), end="")
		
		X = X_train[np.random.permutation(X_train.shape[0])[:batch_size]]#np.random.normal(-4, 1, size=(n, 1))
		Z = np.random.uniform(-1, +1, size=(batch_size, rand_dim))

		fakes = G.predict(Z)

		D.train_on_batch(X, 	yes_labels)
		D.train_on_batch(fakes, no_labels)


		if loss == "wasserstein": 
			# Spectral normalization: 
			# It is not obvious how we should tread convolutions are matrices, for now I 
			# use the approach outlined in the original article.  Maybe if we analyze it using the convolution theorem 
			# we can do it more fine grained and show that this works better? 
			# TODO: The code computes expensive SVD each iteration instead of using power iteration
			# initialized by eigenvector of previous iteration. 
			W1 = D.get_weights()[0] # 3 x 3 x 1 x 64
			W2 = D.get_weights()[6] # 3 x 3 x 64 x 128
			W3 = D.get_weights()[12] # FC ; vector product

			W1 = W1.reshape(64, 	3*3*1) # d_out = 64, 	w=h=3, d_in = 1
			W2 = W2.reshape(128, 3	*3*64) # d_out = 128, 	w=h=3, d_in = 64

			sigma1 = np.linalg.svd(W1, full_matrices=False, compute_uv=False)
			sigma2 = np.linalg.svd(W2, full_matrices=False, compute_uv=False)
			sigma3 = np.linalg.svd(W3, full_matrices=False, compute_uv=False)
			
			W1 = W1 / sigma1.max()
			W2 = W2 / sigma2.max()
			W3 = W3 / sigma3.max()

			weights = D.get_weights()
			weights[0] = W1.reshape(3, 3, 1, 	64)
			weights[6] = W2.reshape(3, 3, 64, 	128)
			weights[12] = W3

		train_G.train_on_batch(Z, 	yes_labels)

		if i % 200 == 0: plot()

		fake_history.append(fakes[:4])

	D.save("models/D_%s_%s.h5"%(loss, clss))
	G.save("models/G_%s_%s.h5"%(loss, clss))
	np.savez("images/dcgan_%s_loss/fake_history_%s"%(loss, clss), fake_history)


if __name__ == "__main__":
	def help(): 
		print("-"*80)
		print("Examples of calling 'dcgan.py'. ")
		print("-"*80)
		print("python dcgan.py vanilla 4")
		print("python dcgan.py vanilla all")
		print("python dcgan.py ls 8")
		print("python dcgan.py ls all")
		print("python dcgan.py wasserstein 9")
		print("python dcgan.py wasserstein all")
		print("python dcgan.py all 3")
		print("python dcgan.py all all")

	losses = ["vanilla", "ls", "wasserstein", "all"]

	if len(sys.argv) == 1: help()
	elif sys.argv[1] not in losses: 
		print("The loss '%s' is not in supported, please use a loss in: "%loss, losses)
		help()

	else: 
		loss = sys.argv[1]
		if len(sys.argv) >= 3: 	clss = sys.argv[2] 
		else:					sys.argv += ["4"] # use 4 as default 

		if loss == "all": 
			for loss in losses.remove("all"): 
				train(loss, clss)
		else: 
			train(loss, clss)


