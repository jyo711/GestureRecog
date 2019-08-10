import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image
from resizeimage import resizeimage
import glob


labels = 5
epochs = 5000
a= 784 # input image size
b = 400 # hidden layer size
display_step = 5


# change the location to file containing all the images locations

with open('/home/abhishek/Documents/BBox-Label-Tool/001_list.txt','r') as fp:
	content = fp.read()
	content  = content.split('\n')
	fp.close()

list2 = []

# extracting file_names of the Labels
file_names = glob.glob("/home/abhishek/Documents/BBox-Label-Tool/Labels/001/*.txt") + glob.glob("/home/abhishek/Documents/BBox-Label-Tool/Labels/002/*.txt") + glob.glob("/home/abhishek/Documents/BBox-Label-Tool/Labels/003/*.txt")+ glob.glob("/home/abhishek/Documents/BBox-Label-Tool/Labels/004/*.txt")
# file_names_2 = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/002")
# file_names_3 = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/003")
# file_names_4 = os.listdir("/home/abhishek/Documents/BBox-Label-Tool/Labels/004")
# all_files = file_names+

#scale = 1
y_ = []
#pa = "home/abhishek/Documents/BBox-Label-Tool/Labels/"

for item in file_names:
	count = 0
	with open(item) as fp: 
		script = fp.read()
		fp.close()
	new_list = script.split('\n')
	y_.append(int(new_list[0]))
	[xmin,ymin,xmax,ymax] = new_list[1].split(" ")
	area = (float(xmin),float(ymin),float(xmax),float(ymax))
	img = Image.open(content[count])
	img = img.convert(mode = "L")
	cropped_img = img.crop(area)
	cover = resizeimage.resize_cover(cropped_img, [28, 28])
	image = np.asarray(cover)
	#print(image)
	#print(image.shape)
	image = np.reshape(image,(1,784))
	list2.append(image)
	#-cv2.imw
	count = count+1

#print(list2[20])

def crop(img, x1, x2, y1, y2, scale):
    crp=img[y1:y2,x1:x2]
    crp=resize(crp,((scale, scale))) 
    return crp

def normalize(R):
        mean = np.mean(R)
        range_val = np.subtract(np.amax(R),np.amin(R),dtype = np.float32)
        R = (R-mean)/float(np.sqrt(np.var(R)))
        return R

#cover.save('test-image-cover.jpeg', image.format)
#print(type(np.asarray(img)))

#def load_img(path):
# 	image = cv2.imread(path)
	
X = np.asarray(list2)
print(X.shape)
X = np.reshape(X,(len(list2),784))
#print(X[2,:])
# for i in range(0,784):
# 	X[:,i] = normalize(X[:,i])
print("shape of X is {}".format(X.shape))
#print(X[2,:])

'''
convert y to logits
'''
y_ = np.eye(labels)[y_]
print("shape of y_ is {}".format(y_.shape))

x = tf.placeholder(tf.float32,shape=(None,a))
y = tf.placeholder(tf.float32,shape=(None,labels))

w1 = tf.Variable(tf.random_normal([a,b]),name = "w1")
b1 = tf.Variable(tf.zeros([b]),name = "b1")

w2 = tf.Variable(tf.random_normal([b,labels]),name = "w2")
b2 = tf.Variable(tf.zeros([labels]),name = "b2")

 	
a1 = tf.matmul(x,w1)+b1
h1 = tf.nn.tanh(a1)
print(h1)

a2 = tf.matmul(h1,w2)+b2
print(a2)
yhat = tf.nn.tanh(a2)
print(yhat)


#accuracy = tf.metrics.accuracy(labels= ,predictions = )

print("Network initalized") 
print("\n")
print("execting training...")
print("\n")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_,logits = yhat))
train_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()	
with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):         
		_,c = sess.run([train_op,loss],feed_dict = {x: X,y: y_})
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "loss={:.9f}".format(c))
		if epoch % 5 == 0:
			save_path = saver.save(sess, "./models/model.ckpt")
			print("Model_{} saved in file: {}".format(epoch,save_path))    
	print("Optimization Finished!")
	correct_prediction = tf.equal(tf.argmax(yhat,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(np.argmax(y_,1))
	print(sess.run(tf.argmax(yhat,1), feed_dict={x: X, y: y_}))
	print(sess.run(accuracy,feed_dict={x:X,y: y_}))