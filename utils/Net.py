import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from resizeimage import resizeimage
import glob
import math
#from sklearn.model_selection import train_test_split

labels = 3
#epochs = 30000
a= 784 # input image size
b = 1024 # hidden layer size
#display_step = 5

def sigmoid(x):
  return 1.0 / (1 + np.exp(-x))

def gesture(list1,frame,w1,w2,b1,b2):
	[xmin,ymin,xmax,ymax] = list1 
	area = (xmin,ymin,xmax,ymax)
	print("area {}".format(area))
	img = Image.fromarray(frame)
	img = img.convert(mode = "L")
	print("size {}".format(img.size))
	cropped_img = img.crop(area)
	print("crp {}".format(cropped_img.size))
	if cropped_img.size != (0,0):
		cover = resizeimage.resize_cover(cropped_img, [28, 28])
		image = np.asarray(cover)
		image = np.reshape(image,(1,784))
		image = (image - 128)/255.0

		# x = tf.placeholder(tf.float32,shape=(None,784))
		# #y = tf.placeholder(tf.float32,shape=(None,labels))

		# w1 = tf.get_variable("w1", shape=[784,1024])
		# b1 = tf.get_variable("b1", shape=[1024])

		# w2 = tf.get_variable("w2", shape=[1024,3])
		# b2 = tf.get_variable("b2", shape=[3])
		 	
		a1 = np.matmul(image,w1)+b1
		h1 = np.tanh(a1)
		#print(h1)

		a2 = np.matmul(h1,w2)+b2
		h2 = np.tanh(a2)

		#print(a2)
		yhat = sigmoid(h2)


		# saver = tf.train.Saver()

		# with tf.Session() as sess1:
		# 	saver.restore(sess1,"/home/abhishek/handtracking/utils/models_l2/model.ckpt")
		# 	print("model_restored")
		print("class: {}".format(yhat))
