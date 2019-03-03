--- ABOUT DATASET ---
Took a 3-4 min full hd file shot with Pixel 2 of me playing guitar. This was rescaled to 32x32 and 64x64 using

	ffmpeg -i raw_data.mp4 -vf scale=32:32 output_32.avi
	ffmpeg -i raw_data.mp4 -vf scale=64:64 output_64.avi 
	(see data/rescle.sh)

The resulting videos are both less than 10 mb thanks to efficient video compression

	data/output_32.avi  # 1.4 MB
	data/output_64.avi  # 4.3 MB

The script 'data/rescale.py' creates numpy files for fast loading (it uses skvideo.io [1])

	data/guitar_32.npz
	data/guitar_64.npz

They also contain a grayscale variant of the video sequence, originally I wanted to train a cycle GAN (the data set has pairs, but the idea was to discard them for the sake of testing cycle GAN). 

[1] http://www.scikit-video.org/stable/io.html


--- 'dcgan.py' ---
The above dataset has shape contains 6914 images of size (32 x 32) or (64 x 64). It seemed a GAN would be easier to train than a cycle GAN, so I started by training a GAN. 

Call examples: 

	dcgan.py 32 
	dcgan.py 64
	dcgan.py (defaults to 32) 

This trains a wasserstein GAN using spectral normalization with power iteration. An image is saved each 200 iteration, see

	images/dcgan_32/
	images/dcgan_64/ 

and the final models are stored in the "models/" directory. 

	models/

Furthermore, the script 'animate.py' creates gifs using a history of fakes from training stored in 'images/dcgan_%i/fake_history.npz'. 

--- 'latent_space_arithmetics.py' ---
# TODO: 
# - Interpolate between different poses. 
# - Is there any meaningful way to do arithmetic? 


--- 'dcgan_cycle.py' ---
# TODO



