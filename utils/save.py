import tensorflow as tf
import numpy as np

w1 = tf.get_variable("w1", shape=[784,400])
b1 = tf.get_variable("b1", shape=[400])

w2 = tf.get_variable("w2", shape=[400,3])
b2 = tf.get_variable("b2", shape=[3])
		 	

saver = tf.train.Saver()
with tf.Session() as sess1:
			saver.restore(sess1,"/home/abhishek/handtracking/utils/models0/model.ckpt")
			print("model_restored")
			np.save("/home/abhishek/handtracking/utils/weight1.npy",w1.eval())
			np.save("/home/abhishek/handtracking/utils/weight2.npy",w2.eval())
			np.save("/home/abhishek/handtracking/utils/bias1.npy",b1.eval())
			np.save("/home/abhishek/handtracking/utils/bias2.npy",b2.eval())