#from darkflow.net.build import TFNet
import cv2
import numpy as np
from utils import detector_utils
import tensorflow as tf
import sys



    
# When everything done, release the capture
# cap.release()
# cv2.destroyAllWindows()

import cv2
import sys
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
# if __name__ == '__main__' :
 
#     # Set up tracker.
#     # Instead of MIL, you can also use
#options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.3}
#tfnet = TFNet(options)
detection_graph, sess = detector_utils.load_inference_graph()
sess = tf.Session(graph=detection_graph)
width = 680
height = 440

def detect_objects(image_np, detection_graph, sess):
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name(
        'detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name(
        'detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name(
        'detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name(
        'num_detections:0')

    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores,
            detection_classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    return np.squeeze(boxes), np.squeeze(scores)

def boundingbox(imgcv,width,height,detection_graph,sess):
        boxes,scores = detect_objects(imgcv,detection_graph,sess)
		# if result is None:
		# 	_,imgcv = video.read()
		# 	continue 
		# bbox = []
        a = return_boxes(2, 0.5, scores, boxes, width, height, imagecv)
				# return (x,y,xw,yh),imgcv	
			# cv2.rectangle(imgcv,(x,y),(xw,yh),(0,255,0),2)
		# cv2.imshow('frame',imgcv)
		# cv2.waitkey(0)
        return ((a[0],a[1]),(a[2],a[3]))
 
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
tracker_type = tracker_types[2]
 
if int(minor_ver) < 3:
    tracker = cv2.Tracker_create(tracker_type)
else:
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
   	    tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
 
# Read video
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, width)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

 
# Exit if video not opened.
if not video.isOpened():
    print "Could not open video"
    sys.exit()
 
# Read first frame.
ok, frame = video.read()
if not ok:
    print 'Cannot read video file'
    sys.exit()
     
# Define an initial bounding box
#bbox = boundingbox(frame)
bbox = boundingbox(imgcv,width,height,detection_graph,sess)
cv2.rectangle(frame,bbox[0],bbox[1],(0,255,0),2)
cv2.circle(frame,(int(bbox[0][0]-bbox[1][0]),int(bbox[0][1]-bbox[1][1])),10,(255,255,255),-11)
cv2.imshow("Tracking", frame)
cv2.waitKey(0)

# print bbox
xtl = bbox[1][0]
ytl = bbox[1][1]
w = np.abs([1][0]-bbox[0][0])/4
h = np.abs(bbox[1][1]-bbox[0][1])/4
print xtl,ytl,w,h
# bbox = (287, 23, 86, 320)
# bbox = (206, 14,678, 472)
# # video.release()
# # cv2.destroyAllWindows()
 
#     # Uncomment the line below to select a different bounding box
#     # bbox = cv2.selectROI(frame, False)
 
#     # Initialize tracker with first frame and bounding box
# print 
ok = tracker.init(frame,(xtl,ytl,w,h))
 
while True:
        # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
         
        # Start timer
    timer = cv2.getTickCount()
 
        # Update tracker
    ok, bbox = tracker.update(frame)
 
        # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
    if ok:
            # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
            # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
 
        # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
     
        # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
        # Display result
    cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break











# imgcv = cv2.imread("./sample_img/sample_person.jpg")
# result = tfnet.return_predict(imgcv)

# for i in range(len(result)):
# 	x,y = result[i]['bottomright']['x'],result[i]['bottomright']['y']
# 	xw,yh = result[i]['topleft']['x'], result[i]['topleft']['y']
# 	print (x,y,xw,yh)
# 	cv2.rectangle(imgcv,(x,y),(xw,yh),(0,255,0),2)
# cv2.imshow('frame',imgcv)
# cv2.waitKey(0)
# print(result)

