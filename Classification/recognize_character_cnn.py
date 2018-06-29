import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Classification.CNN.classify import *

# find the most likely character by template matching
def recognize_character(char):
    print("Recognizing the following character:")
    #kernel = np.ones((3,3),np.uint8)
    #char = cv2.erode(char, kernel, iterations = 1)
    user_defined_threshold = 0.08

    top_chars, top_scores = recognition_char(char, user_defined_threshold)

    return top_chars, top_scores


