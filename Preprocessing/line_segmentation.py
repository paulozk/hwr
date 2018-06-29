import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from scipy.ndimage import rotate
from Preprocessing.word_segmentation import segment_words



# extract lines/sentences from the parchment, based on the average height of the boxes
def segment_line_strips(boxes, box_centroids, parchment, avg_height, avg_width):
    img_to_find_background = parchment.copy()
    k_size_y = int(avg_height / 2)
    k_size_x = int(avg_width / 2)
    k2_size_y = int(avg_height / 5)
    k2_size_x = int(avg_width / 5)
    kernel = np.ones((k_size_y, k_size_x), np.uint8)
    kernel2 = np.ones((k2_size_y + 1, k2_size_x + 1), np.uint8)
    kernel3 = np.ones((k2_size_y, k2_size_x), np.uint8)

    img_to_find_background = cv2.morphologyEx(img_to_find_background, cv2.MORPH_CLOSE, kernel2)
    img_to_find_background = cv2.morphologyEx(img_to_find_background, cv2.MORPH_OPEN, kernel3)
    img_to_find_background = cv2.morphologyEx(img_to_find_background, cv2.MORPH_CLOSE, kernel)
    # parchment = 255 - parchment
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_to_find_background, 8, cv2.CV_32S)

    trg_label = 0
    for i in range(len(stats)):
        if stats[i][0] == 0 and stats[i][1] == 0:
            trg_label = i
    # fuse the black background with the white parchment
    # parchment = 255 - parchment
    parchment[labels == trg_label] = 255
    # _,parchment = cv2.threshold(parchment, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    height, width = parchment.shape    
    parchment, na, na2 = rotation_procedure(parchment.copy(), avg_height)
    line_image = parchment.copy()
    
    boxes, centroids, avg_height, avg_width = segment_words(parchment)
    
    box_threshold  = avg_height * 2
    line_count_threshold = 2

    word_lines = []
    line = []

    # calculate average centroids for each line
    line.append(boxes[0][6])
    line.append(1)
    word_lines.append(line)
    for temp_i in range(len(boxes) - 1):
        i = temp_i + 1
        box = boxes[i]
        #check if box belongs to the same line as the previous one: check if centroid of box falls within the range of the previous
        #box
        centroid_y = box[6]
        box_height = box[3]

        if box_height < box_threshold:
            empty_flag = True
            for temp_line in word_lines:
                temp_centroid = temp_line[0]
                if(centroid_y >= (temp_centroid - avg_height) and (centroid_y <= temp_centroid + avg_height)):
                    empty_flag = False
                    temp_line[0] = (temp_line[0] * temp_line[1] + centroid_y) / (temp_line[1] + 1)
                    temp_line[1] += 1
            if empty_flag:
                line = []
                line.append(box[6])
                line.append(1)
                word_lines.append(line)

    for line in word_lines:
        if line[1] < line_count_threshold:
            line[0] = -100



    # include boxes within certain height of the average centroids
    for i in range(len(boxes)):
        box = boxes[i]
        #check if box belongs to the same line as the previous one: check if centroid of box falls within the range of the previous
        #box
        centroid_y = box[6]
        box_height = box[3]

        if box_height < box_threshold:
            empty_flag = True
            for temp_line in word_lines:
                temp_centroid = temp_line[0]
                if(centroid_y >= (temp_centroid - avg_height) and (centroid_y <= temp_centroid + avg_height)):
                    temp_line.append(box)

    for line in word_lines:
        line.pop(0)
        line.pop(0)
        
        
        # to preserve more line spacebased on average_height
    line_buffer_fraction = 0.1
    line_buffer = int(line_buffer_fraction * avg_height)

    #save strips
    strips = []

    # draw lines for the first strip
    for line in word_lines:
        if len(line) > 0:
            min_height = 999999
            max_height = 0
            for box in line:
                if(box[1] < min_height):
                    min_height = box[1]
                if(box[1] + box[3]  > max_height):
                    max_height = box[1] + box[3] 

            min_height = int(min_height)
            max_height = int(max_height)

            cv2.line(line_image,(0, min_height),(width, min_height),(0, 200,0), 4)   
            cv2.line(line_image,(0, max_height),(width, max_height),(0, 200,0), 4) 

            min_height = 0 if (min_height - line_buffer) < 0 else (min_height - line_buffer)
            x=True if 'a'=='a' else False
            max_height = height if (max_height + line_buffer) > height else (max_height + line_buffer)
            
            #cv2.line(parchment,(0, min_height),(width, min_height),(0, 200,0), 4)   
            
            strips.append([min_height, max_height])
            
    #plt.figure(figsize = (500,10))
    #plt.imshow(parchment, cmap='gray', aspect = 1)
    #plt.show()
    return strips, parchment, line_image

def find_peak(hist, peak_window=5, grey_threshold=100):
    peak_kernel = peak_window // 2
    rg = range(len(hist))
    list_peaks = []
    for i in rg:
        max_count = 0
        for j in range(i, i + peak_window):
            if max_count < hist[i]:
                max_count = hist[i]

        # find if the max is peak
        peak_flag = True
        for k in range(peak_kernel):
            idx_diff = k + 1
            idx_min = 0 if (i - idx_diff) < 0 else i - idx_diff
            idx_max = (len(hist)-1) if (i + idx_diff) > (len(hist)-1) else i + idx_diff
            if max_count <= hist[idx_min] or max_count <= hist[idx_max]:
                peak_flag = False
        if peak_flag:
            list_peaks.append([i, max_count])
#     list_peaks.sort(key=lambda x: x[1], reverse=True)
    
    return list_peaks

def rotate_img(img, degree=10, interval=1):
    degree_start = abs(degree)
    degree_end = -(degree_start + 1)
    interval = int(interval)
    if interval < 1:
        raise ValueError(
                "parameter interval should be at least 1, now it's {}".format(interval))
    
    images_to_return = []
    for i in range(degree_start, degree_end, -interval):
        processed = img.copy()    
        processed = rotate(processed, i, reshape=True, cval=255)
        images_to_return.append(processed)
    return images_to_return
            
def rotation_procedure(image, avg_height):
    rotated_img_list = rotate_img(image, 10, 1)
    best_img = image.copy()
    line_strips = []
    max_range = 0    
    for rotated_img in rotated_img_list:
        
      # verticalProfile = np.sum(whitened_parchment, axis=0)
        horizontalProfile = np.sum(rotated_img, axis=1)        
        temp_max_range = horizontalProfile.max() - horizontalProfile.min()
        
        if temp_max_range > max_range:
            max_range = temp_max_range
            best_img = rotated_img
            best_horizontalProfile = horizontalProfile
            # plt.plot(range(0,columns), verticalProfile)

#     [rows, columns] = best_img.shape
#     plt.plot(best_horizontalProfile, range(0,rows))
#     plt.show()
    # adjust window size
    peak_list = find_peak(best_horizontalProfile, int(avg_height))
    copy_img = best_img.copy()
    height, width = best_img.shape
    min_height = 0
    for peak_idx in range(len(peak_list)):
        if peak_idx == 0:
            min_height = peak_list[peak_idx][0]
        else:
            max_height = peak_list[peak_idx][0]
            line_strips.append([min_height, max_height])
            cv2.line(copy_img,(0, min_height),(width, min_height),(0, 200,0), 4)   
            cv2.line(copy_img,(0, max_height),(width, max_height),(0, 200,0), 4) 
            min_height = max_height

#     plt.figure(figsize = (500,10))
#     plt.imshow(copy_img, cmap='gray', aspect = 1)
#     plt.show()    
    
    return best_img, line_strips, copy_img
