"""
	Compares the speed of different approaches to computing largest eigenvalue. 
	Quite surprisingly the NumPy variant performed the fastest! 

	The plot in 'test_speed.pdf' display some weird behaviour for the tensorflow 
	implementation, maybe moving stuff back and forth from GPU takes time?
	For now the NumPy variant is fast enough for the w-gan. 

	TODO: Figure out if there is anything wrong with the GPU implementation. 
	Also, when training the w-gan everything all the matrices lives in the GPU
	RAM, so figuring out what is wrong on the GPU aught to make matters faster. 

"""

import numpy as np
import tensorflow as tf
from power_iteration import *
import matplotlib.pyplot as plt
import time

fig, ax = plt.subplots(1, 2)
sess = tf.Session()

time_svd = []
time_np = []
time_tf = []

estimate_svd = []
estimate_np = []
estimate_tf = []

iterations = 30

for i in range(1, iterations, 1):
	print("--- [%i / %i] ---"%(i, iterations))

	size = i*100

	a = np.random.normal(0, 1, size=(size, size)).astype(np.float32)

	t0 = time.time()
	estimate_svd.append(np.max(sess.run(tf.linalg.svd(a)[0])))
	t1 = time.time()
	estimate_np.append(power_iteration_np(a, 50))
	t2 = time.time()
	estimate_tf.append(power_iteration_tf(a, sess, 50)[0][0])
	t3 = time.time()

	time_svd.append(t1-t0)
	time_np.append(t2-t1)
	time_tf.append(t3-t2)

	for j in range(2): ax[j].cla()
	ax[0].set_title("Time")
	ax[0].plot(time_svd, label="TF: SVD") # do np svd?
	ax[0].plot(time_np, label="NP: Power Iteration")
	ax[0].plot(time_tf, label="TF: Power Iteration")

	ax[1].set_title("Estimate")
	ax[1].plot(estimate_svd, label="SVD")
	ax[1].plot(estimate_np, label="NP: Power Iteration")
	ax[1].plot(estimate_tf, label="TF: Power Iteration")
	for j in range(2): ax[j].legend()

	plt.tight_layout()
	plt.pause(.1)

plt.savefig("test_speed.pdf")
