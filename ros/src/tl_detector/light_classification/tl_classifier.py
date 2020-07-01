from styx_msgs.msg import TrafficLight
import cv2
import numpy as np
import rospy
import tensorflow as tf
import os
import time

class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.counter = 0        
        self.light_class = TrafficLight.RED
        rospy.logwarn("current path")
        rospy.logwarn(os.path.abspath(os.getcwd()))
        SSD_GRAPH_FILE = './light_classification/frozen_inference_graph.pb'
        self.classes = []
        self.boxes = []
        
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(SSD_GRAPH_FILE, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')    
        self.sess = tf.Session(graph=self.detection_graph) 
                
    def get_classification(self, image):        
        self.counter = self.counter + 1
        if self.counter < 5:
            rospy.logwarn("tick")
            return self.light_class
        
        self.counter = 0
        
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        start = time.time()
        with self.detection_graph.as_default():                
            # Actual detection.
            (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes], 
                                                feed_dict={self.image_tensor: image_np})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            min_score = 0.2
            # Filter boxes with a confidence score less than `confidence_cutoff`
            # boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            #width, height = image.size
            #box_coords = to_image_coords(boxes, height, width)

            # Each class with be represented by a differently colored box
            n = len(classes)
            idxs = []            
            for i in range(n):                
                if scores[i] >= min_score and classes[i]==10:
                    idxs.append(i)

            self.boxes = boxes[idxs, ...]            
            self.classes = classes[idxs, ...]
        end = time.time()
        rospy.logwarn("Elapsed time: " + str(end - start))        
        
        rospy.logwarn(self.classes)                
        rospy.logwarn(self.boxes)                
        #TODO implement light color prediction
        height,width,channels = image.shape
        
        y1 = 0
        y2 = height

        x1 = 0
        x2 = width
        
        #rospy.logwarn("x1: " +str(x1) + " x2: " + str(x2) + " y1: " + str(y1) + " y2: " + str(y2))
        
        if self.boxes is not None:
            if self.boxes.shape[0] > 0:            
                rospy.logwarn("van adat")
                y1 = int(600 * self.boxes[0][0])
                y2 = int(600 * self.boxes[0][2])
                x1 = int(800 * self.boxes[0][1])
                x2 = int(800 * self.boxes[0][3])

        image = image[y1:y2,x1:x2]
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_red = np.array([0,150,50])
        upper_red = np.array([4,255,255])
        mask0 = cv2.inRange(image_hsv, lower_red, upper_red)

        lower_red = np.array([175,150,50])
        upper_red = np.array([180,255,255])
        mask1 = cv2.inRange(image_hsv, lower_red, upper_red)

        lower_green = np.array([55,150,50])
        upper_green = np.array([65,255,255])
        mask_green = cv2.inRange(image_hsv, lower_green, upper_green)

        mask_red = mask0 + mask1
        
        output_img_red = image.copy()
        output_img_red[np.where(mask_red==0)] = 0

        output_img_green = image.copy()
        output_img_green[np.where(mask_green==0)] = 0

        reds = len(np.nonzero(output_img_red)[0])
        greens = len(np.nonzero(output_img_green)[0])
        rospy.logwarn("reds: " +str(reds) + "greens: " + str(greens))
        
        if reds>greens and reds>100:
            rospy.logwarn("RED!")
            self.light_class = TrafficLight.RED
        
        elif reds<greens and greens>100:
            rospy.logwarn("GREEN!")
            self.light_class = TrafficLight.GREEN                    
        else:
            self.light_class = TrafficLight.UNKNOWN                    
            
        return self.light_class
