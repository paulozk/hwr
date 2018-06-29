import tensorflow as tf
import sys
import os
import tkinter as tk
import numpy as np
import cv2
from tkinter import filedialog

# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

root = tk.Tk()
root.withdraw()


# def to_rgb1b(im):
#     # I would expect this to be identical to 1a
#     w, h = im.shape
#     ret = np.empty((w, h, 3), dtype=np.uint8)
#     ret[:, :, 0] = im
#     ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
#     return ret



# if image_path:
#
#     # Read the image_data
#     image_data = tf.gfile.FastGFile(image_path, 'rb').read()
#
# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line in tf.gfile.GFile("Classification/CNN/tf_files/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("Classification/CNN/tf_files/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def recognition_char(image, user_defined_threshold):
    # image as numpy array
    # image = to_rgb1b(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = tf.convert_to_tensor(image, np.float32)
    image = np.array(image)[:, :, :]

    # image_path = sys.argv[1]
    # image_path = filedialog.askopenfilename()

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        # predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})  # this reads tensor image from disk
        predictions = sess.run(softmax_tensor, {'DecodeJpeg:0': image})  # this reads tensor image from np array

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        list_candidates = []
        list_score = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            if score >= user_defined_threshold:
                list_candidates.append(human_string)
                list_score.append(score)
            # print('%s (score = %.5f)' % (human_string, score))

        return list_candidates, list_score


# C:\Users\Andrew X\Documents\GitHub\handwriting_recognition\Preprocessing\CNN_test\test2.jpg
# image_path = 'C:/Users/Andrew X/Documents/GitHub/handwriting_recognition/Preprocessing/CNN_test/test2.jpg'
# image = image_data = tf.gfile.FastGFile(image_path, 'rb').read()
# recognition_char(image)
