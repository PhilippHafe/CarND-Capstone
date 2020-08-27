from styx_msgs.msg import TrafficLight
import tensorflow as tf
import numpy as np
import datetime

class TLClassifier(object):
    def __init__(self):
        PATH_TO_CKPT = "light_classification/frozen_inference_graph_josehoras.pb"
        self.graph = tf.Graph()
        self.threshold = 0.5

        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)        
                tf.import_graph_def(od_graph_def, name = '')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        with self.graph.as_default():
            image_np_expanded = np.expand_dims(image, axis=0)
            start = datetime.datetime.now()
            (boxes, scores, classes, num) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_np_expanded})
            end = datetime.datetime.now()
            dur = end - start
            #print(dur.total_seconds())

        boxes = np.squeeze(boxes)
        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        
	if len(scores)>0 and np.max(scores)>self.threshold:
	    detected_class = int(classes[np.argmax(scores)])
	else:
	    detected_class = 4

        if detected_class == 1:
            print('Classes: {}, Green. Detection Duration {}'.format(TrafficLight.GREEN,dur.total_seconds()))
            return TrafficLight.GREEN
        elif detected_class == 3:
            print('Classes: {}, Red Detection Duration {}'.format(TrafficLight.RED,dur.total_seconds()))
            return TrafficLight.RED
        elif detected_class == 2:
            print('Classes: {}, Gelb. Detection Duration {}'.format(TrafficLight.YELLOW,dur.total_seconds()))
            return TrafficLight.YELLOW

        return TrafficLight.UNKNOWN
