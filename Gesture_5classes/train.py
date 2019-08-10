import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from resizeimage import resizeimage

with open('001_list.txt','r') as fp:
	content = fp.read()
	content  = content.split('\n')
	fp.close()

list2 = []

# extracting file_names of the labels
file_names = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/001") + os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/002") + os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/003")+ os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/004")
# file_names_2 = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/002")
# file_names_3 = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/003")
# file_names_4 = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/004")
# all_files = file_names+

scale = 1
y_ = []

for item in file_names:
	count = 0
	with open(item) as fp: 
		script = fp.read()
		fp.close()
	new_list = script.split('\n')
	y_.append(new_list[0])
	xmin,ymin.xmax,ymax = new_list[1].split(' ')
	area = (xmin,ymin,xmax,ymax)
	img = Image.open(content[count])
	cropped_img = img.crop(area)
	cover = resizeimage.resize_cover(cropped_img, [28, 28])
    image = np.asarray(cover)
    image = np.reshape(img,(1,784))
    list2.append(image)
	#-cv2.imw
	count = count+1

def crop(img, x1, x2, y1, y2, scale):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((scale, scale))) 
    return crp

#cover.save('test-image-cover.jpeg', image.format)
#print(type(np.asarray(img)))

#def load_img(path):
# 	image = cv2.imread(path)
	
X = np.asmatrix(list2)
X = np.reshape(x,(len(list2),784))

'''
convert y to logits
'''
y_ = tf.onehot(indices=y_,depth=4)


x = tf.placeholder(shape=[None,a])
y = tf.placeholder(shape=[None,4])

w1 = tf.Variable(shape = [a,400])
b1 = tf.Variable(shape = [400,1])

w2 = tf.Variable(shape = [400,4])
b2 = tf.Variable(shape = [4,1])

 	
a1 = tf.matmul(x,w1)+b1
h1 = tf.sigmoid(a1)

a2 = tf.matmul(h1,w2)+b2
yhat = tf.sigmiod(a2)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = yhat))
train_op = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(loss)

init = tf.gloabal_variable_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run()

	for epoch in range(epochs):         
		_,c = sess.run([train_op,loss],feed_dict = {x: X,y = y_})
		if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(c))
    print("Optimization Finished!")
	save_path = saver.save(sess, "/tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)