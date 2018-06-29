import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from scipy.ndimage import rotate


# find words and characters in the parchment and compute the average height and width
def segment_words(parchment):
    #plt.figure(figsize = (500,4))
    #plt.imshow(parchment, cmap='gray', aspect = 1)
    #plt.show()
    
    # remove some noise from the sauvola binarized parchment
    kernel = np.ones((5,5),np.uint8)
    image_bin = cv2.morphologyEx(parchment, cv2.MORPH_CLOSE, kernel)
    image_bin = 255 - image_bin
    # find connected components (words) from the parchment
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin, 8, cv2.CV_32S)
    image_bin = 255 - image_bin
    # get the components that are likely to be words/characters
    min_thresh = 600
    max_thresh = 10000
    boxes = []
    box_centroids = []
    for i in range(len(stats)):
        if(stats[i][4] >= min_thresh and stats[i][4] <= max_thresh):
            x = stats[i][0]
            y = stats[i][1]
            width = stats[i][2]
            height = stats[i][3]
            temp_stat = np.append(stats[i], centroids[i])
            boxes.append(temp_stat)
            box_centroids.append(centroids[i])
            #cv2.rectangle(image,(x,y),(x + width,y + height),(0,200,0),3)
    avg_width = 0
    avg_height = 0
    N = len(boxes)
    
    for box in boxes:
        avg_width += box[2]
        avg_height += box[3]
    avg_width /= N
    avg_height /= N
    
  
    
    
    
    return boxes, box_centroids, avg_height, avg_width