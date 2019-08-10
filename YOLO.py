import tensorflow as tf
import numpy as np

images = tf.reshape(features["x"],[-1,32,32,3])
flag = mode == tf.estimator.ModeKeys.TRAIN
init =args.init	
if init ==-1:
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
else:
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

 
#check the input image size for padding
 
 
def net(images):
 	conv1 = tf.layers.conv2d(inputs=images,filters=32,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c1_bn = tf.layers.batch_normalization(inputs=conv1,axis=-1,momentum=0.99,epsilon=0.001,training=flag) 

 	pool1 = tf.layers.max_pooling2d(inputs=c1_bn, pool_size=[2, 2], strides=2)
 	
 	conv2 = tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c2_bn = tf.layers.batch_normalization(inputs=conv2,axis=-1,momentum=0.99,epsilon=0.001,training=flag)

 	pool2 = tf.layers.max_pooling2d(inputs=c2_bn, pool_size=[2, 2], strides=2) # 56*56

 	conv3 = tf.layers.conv2d(inputs=pool2,filters=128,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c3_bn = tf.layers.batch_normalization(inputs=conv3,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv4 = tf.layers.conv2d(inputs=c3_bn,filters=64,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c4_bn = tf.layers.batch_normalization(inputs=conv4,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv5 = tf.layers.conv2d(inputs=conv4,filters=128,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c5_bn = tf.layers.batch_normalization(inputs=conv5,axis=-1,momentum=0.99,epsilon=0.001,training=flag)

 	pool3 = tf.layers.max_pooling2d(inputs=c5_bn, pool_size=[2, 2], strides=2)

 	conv6 = tf.layers.conv2d(inputs=pool3,filters=256,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c6_bn = tf.layers.batch_normalization(inputs=conv6,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv7 = tf.layers.conv2d(inputs=c6_bn,filters=128,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c7_bn = tf.layers.batch_normalization(inputs=conv7,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv8 = tf.layers.conv2d(inputs=c7_bn,filters=256,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c8_bn = tf.layers.batch_normalization(inputs=conv8,axis=-1,momentum=0.99,epsilon=0.001,training=flag)

 	pool4 = tf.layers.max_pooling2d(inputs=conv8, pool_size=[2, 2], strides=2)

 	conv9 = tf.layers.conv2d(inputs=pool4,filters=512,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
	c9_bn = tf.layers.batch_normalization(inputs=conv9,axis=-1,momentum=0.99,epsilon=0.001,training=flag) 	
 	conv10 = tf.layers.conv2d(inputs=c9_bn,filters=256,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
	c10_bn = tf.layers.batch_normalization(inputs=conv1,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
	conv11 = tf.layers.conv2d(inputs=c10_bn,filters=512,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
	c11_bn = tf.layers.batch_normalization(inputs=conv11,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
	conv12 = tf.layers.conv2d(inputs=conv11,filters=256,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c12_bn = tf.layers.batch_normalization(inputs=conv12,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv13 = tf.layers.conv2d(inputs=c12_bn,filters=512,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c13_bn = tf.layers.batch_normalization(inputs=conv13,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	
 	pool5 = tf.layers.max_pooling2d(inputs=c13_bn, pool_size=[2, 2], strides=2)
 	
 	conv14 = tf.layers.conv2d(inputs=pool5,filters=1024,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c14_bn = tf.layers.batch_normalization(inputs=conv14,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv15 = tf.layers.conv2d(inputs=c14_bn,filters=512,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c15_bn = tf.layers.batch_normalization(inputs=conv15,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv16 = tf.layers.conv2d(inputs=c15_bn,filters=1024,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c16_bn = tf.layers.batch_normalization(inputs=conv16,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv17 = tf.layers.conv2d(inputs=c16_bn,filters=512,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c17_bn = tf.layers.batch_normalization(inputs=conv17,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv18 = tf.layers.conv2d(inputs=c17_bn,filters=1024,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
 	c18_bn = tf.layers.batch_normalization(inputs=conv18,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
 	conv19 = tf.layers.conv2d(inputs=c18_bn,filters=1000,kernel_size=[1,1],padding="same",activation=tf.nn.relu,kernel_initializer = initializer)
    c19_bn = tf.layers.batch_normalization(inputs=conv19,axis=-1,momentum=0.99,epsilon=0.001,training=flag)
  
 
 