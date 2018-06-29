import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from scipy.ndimage import rotate

def extract_parchment(image, grey_thre):
    output_image = np.array(image, copy=True)  
    
    # set dark grey to black
    ret, thresh_image = cv2.threshold(image, grey_thre, 255, cv2.THRESH_TOZERO)
    thresh_image = cv2.GaussianBlur(thresh_image,(5,5),0)
    
    # closing
    kernel = np.ones((5,5),np.uint8)
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    
    # connected component
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image, 8, cv2.CV_32S)
    area = stats[:,4]
    
    # filter small components
    filtered_components = []
    threshold = 8000
    for i in range(len(area)):
        if(area[i] > threshold and not(stats[i][0] == 0 and stats[i][1] == 0)): # remove background component of (0, 0)
            filtered_components.append(i)
            
    # find best component
    height, width = thresh_image.shape
    biggest_area = 0
    best_component = 0    
    for component in filtered_components:
        if(area[component] > biggest_area):
            biggest_area = area[component]
            best_component = component 
            
    # Set selected component to white, everything else to black
    for i in range(height):
        for j in range(width):
            if(labels[i][j] == best_component):
                thresh_image[i][j] = 255
            else:
                thresh_image[i][j] = 0
                output_image[i][j] = 0
                
                
    # closing again
    thresh_image = cv2.GaussianBlur(output_image,(5,5),0)
    kernel = np.ones((80,80),np.uint8)
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_CLOSE, kernel)
    dilation_kernel = np.ones((50,50),np.uint8)
    thresh_image = cv2.dilate(thresh_image,dilation_kernel,iterations = 1)
    # connected component
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image, 8, cv2.CV_32S)
    area = stats[:,4]
    
    # filter small components
    filtered_components = []
    threshold = 8000
    for i in range(len(area)):
        if(area[i] > threshold and not(stats[i][0] == 0 and stats[i][1] == 0)): # remove background component of (0, 0)
            filtered_components.append(i)
            
    # find best component
    height, width = thresh_image.shape
    biggest_area = 0
    best_component = 0    
    for component in filtered_components:
        if(area[component] > biggest_area):
            biggest_area = area[component]
            best_component = component 
            
    # Set selected component to white, everything else to black
    for i in range(height):
        for j in range(width):
            if(labels[i][j] == best_component):
                pass
            else:
                image[i][j] = 0
    
    
    parchment_stats = stats[best_component,:]
    x = parchment_stats[0]
    y = parchment_stats[1]
    width = parchment_stats[2]
    height = parchment_stats[3]
    parchment = image[y:y+height, x:x+width]
    
    # apply sauvola binarization to the threshold
    parchment = binarization(parchment)
    parchment.dtype='uint8'
    parchment[(parchment > 0)] = 255

    return parchment

def binarization(parchment):
    blurred = cv2.GaussianBlur(parchment,(151,151),0)
    window_size = 25
    thresh_sauvola = threshold_sauvola(blurred, window_size=window_size)
    binary_sauvola = parchment > thresh_sauvola
  
    return binary_sauvola