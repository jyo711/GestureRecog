import cv2
import numpy as np
import sys
import cv2

def iou(r1,r2):
    intersect_w = np.maximum(np.minimum(r1[0]+r1[2],r2[0]+r2[2])-np.maximum(r1[0],r2[0]),0)
    intersect_h = np.maximum(np.minimum(r1[1]+r1[3],r2[1]+r2[3])-np.maximum(r1[1],r2[1]),0)
    area_r1 = r1[2]*r1[3]
    area_r2 = r2[2]*r2[3]
    intersect = intersect_w*intersect_h 
    union = area_r1 + area_r2 - intersect
    
    return intersect/union

def tracker_init():
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.') 
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
    return tracker        
 
def  run_tracker(tracker,box,fram,flag):
    print("box {}".format(box))
    xtl = box[0]
    ytl = box[1]
    w = np.abs(box[2]-box[0])/2
    h = np.abs(box[3]-box[1])/2

    # print 
    if flag == False:   
        ok = tracker.init(fram,(xtl,ytl,w,h))
    #print("starting")
    #count = 0 
    while True:
        timer = cv2.getTickCount()

            # Update tracker
        
        ok, bbox = tracker.update(fram)
        print("bbox:{}".format(bbox))
        if ok:
                # Tracking success
            #flag = True    
            p1 = (int(bbox[0]),int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(fram, p1, p2, (255,0,0), 2, 1)
        else :
                # Tracking failure
            print("tracking failure")
            return False

        cv2.imshow("Tracking", fram)
        #count = count+1 
            # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        #if k == 27 : break
        return True



