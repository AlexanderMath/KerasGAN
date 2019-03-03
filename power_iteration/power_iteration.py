"""
	Thanks to taki0112 for sharing an implementation of spectral normalization under the MIT license. 

		https://github.com/taki0112/Spectral_Normalization-Tensorflow

	The code below is a slight modification the NumPy code from Wikipedia 

		https://en.wikipedia.org/wiki/Power_iteration
		https://en.wikipedia.org/wiki/Rayleigh_quotient (compute eigenvalue given eigenvect)

	The NumPy code was rewritten to Tensorflow inspired by taki0112. 

	The power_iteration methods below computes the largest spectral value, that is, the
	square root of the largest eigenvalue of A.T @ A, see: 

		https://en.wikipedia.org/wiki/Singular_value

"""
import numpy as np
import tensorflow as tf
import time


# --- NUMPY CODE ---
def l2_normalize(b): return b / np.linalg.norm(b)

def power_iteration_np(A, num_simulations=10, init_vect=None, approximate_scale=1.00, return_vector=False):
	A = A.T @ A # TODO: IS THERE ANY DIFFERENCE TO A @ A.T? IF NOT, USE THE SMALLEST!
	if init_vect is None: 	b_k = np.random.rand(A.shape[1])
	else:					b_k = init_vect

	for k in range(num_simulations): 	b_k 	= l2_normalize(A @ b_k )
 
	if return_vector: 	return np.sqrt(b_k.T @ A @ b_k / (b_k.T @ b_k  )) * approximate_scale, b_k  
	else:			return np.sqrt(b_k.T @ A @ b_k / (b_k.T @ b_k  )) * approximate_scale
# --- END NUMPY CODE ---


# --- TF CODE --- 
def power_iteration_tf(A, sess, num_simulations=10, init_vect=None, approximate_scale=1.00, return_vector=False):
	A = tf.transpose(A) @ A
	if init_vect is None: 	b_k = tf.get_variable("b_k"+str(time.time()), (A.shape[0], 1), initializer=tf.random_normal_initializer(), trainable=False)
	else:					b_k = init_vect

	sess.run(b_k.initializer)

	for k in range(num_simulations):	 b_k = tf.nn.l2_normalize( A @ b_k )
 
	if return_vector: 	return np.sqrt(sess.run(tf.transpose(b_k) @ A @ b_k / (tf.transpose(b_k) @ b_k))) * approximate_scale, sess.run(b_k)
	else: 			return np.sqrt(sess.run(tf.transpose(b_k) @ A @ b_k / (tf.transpose(b_k) @ b_k))) * approximate_scale
# --- END TF CODE --- 

# --- RENAME FASTEST (see test_speed.py for experiment) --- 
def power_iteration(A, num_simulations=10, init_vect=None, approximate_scale=1.00, return_vector=False): 
	return power_iteration_np(A, num_simulations, init_vect, approximate_scale, return_vector)

if __name__ == "__main__": 

	a = np.random.normal(0, 1, size=(100, 100)).astype(np.float32)

	sess = tf.Session()
	print("SVD: \t%.8f"% 	np.max(sess.run(tf.linalg.svd(a)[0])))
	print("NumPy: \t%.8f"% 	power_iteration_np(a, 50))
	print("TF: \t%.8f"% 	power_iteration_tf(a, sess, 50)[0][0]) 


