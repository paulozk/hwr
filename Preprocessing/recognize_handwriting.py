import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

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
   
    parchment = 255 - parchment
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(parchment, 8, cv2.CV_32S)
    # fuse the black background with the white parchment
    parchment = 255 - parchment
    parchment[labels == 1] = 255
    print(stats)
    #_,parchment = cv2.threshold(parchment, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return parchment

def binarization(parchment):
    blurred = cv2.GaussianBlur(parchment,(151,151),0)
    window_size = 25
    thresh_sauvola = threshold_sauvola(blurred, window_size=window_size)
    binary_sauvola = parchment > thresh_sauvola
  
    return binary_sauvola

# find words and characters in the parchment and compute the average height and width
def segment_words(parchment):
    kernel = np.ones((5,5),np.uint8)
    # remove some noise from the sauvola binarized parchment
    plt.figure(figsize = (500,4))
    plt.imshow(parchment, cmap='gray', aspect = 1)
    plt.show()
    image_bin = cv2.morphologyEx(parchment, cv2.MORPH_CLOSE, kernel)
    image_bin = 255 - image_bin
    # find connected components (words) from the parchment
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(image_bin, 8, cv2.CV_32S)
    image_bin = 255 - image_bin
    # get the components that are likely to be words/characters
    min_thresh = 400
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

# extract lines/sentences from the parchment, based on the average height of the boxes
def segment_line_strips(boxes, box_centroids, parchment, avg_height, avg_width):
    height, width = parchment.shape
    line_image = np.array(parchment)
    """word_lines = []
    line = []
    # add first box to first line
    line.append(boxes.pop(0))
    previous_centroid = box_centroids[0][1]
    box_centroids.pop(0)
    for i in range(len(boxes)):
        box = boxes[i]
        #check if box belongs to the same line as the previous one: check if centroid of box falls within the range of the previous
        #box
        centroid_y = box_centroids[i][1]
        if(centroid_y >= (previous_centroid - avg_height) and (centroid_y <= previous_centroid + avg_height)):
            line.append(box)
            previous_centroid = centroid_y
        else:
            #if not, add the line to the collection of lines and start a new one
            if(line != []):
                word_lines.append(line)
                line = []
                previous_centroid = centroid_y
                line.append(box)
    #save strip images
    strips = []
    # draw lines for the first strip
    for line in word_lines:
        min_height = 99999
        max_height = 0
        for box in line:
            if(box[1] < min_height):
                min_height = box[1]
            if(box[1] + box[3]  > max_height):
                max_height = box[1] + box[3] 
        #cv2.line(line_image,(0,min_height),(width, min_height),(0,200,0),3)   
        #strips.append([min_height, max_height])
        strips.append(parchment[min_height:max_height, :])
    
    return strips"""
    
    box_threshold  = avg_height * 2
    line_count_threshold = 1

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
        print(line)


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
            strips.append([min_height, max_height])
    return strips

def extract_words(strip):
    kernel = np.ones((5,5),np.uint8)
    strip_morphed = 255 - strip
    strip_morphed = cv2.dilate(strip_morphed,kernel,iterations = 2)
    #strip = cv2.GaussianBlur(strip, (15,15), 10)
    #strip = cv2.GaussianBlur(strip, (35,35), 10)
    _, strip_morphed = cv2.threshold(strip_morphed, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(strip_morphed, 8, cv2.CV_32S)
    
    """plt.figure(figsize = (500,4))
    plt.imshow(strip, cmap='gray', aspect = 1)
    plt.show()"""
    strip_morphed = 255 - strip
    stats = stats[1:]
    words = []
    # get all components (words), except the first one (background) 
    # (x, y, width, height)
    for stat in stats:
        if(stat[4] > 1000):
            word = strip[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]]
            words.append(word)
            """plt.figure(figsize = (500,4))
            plt.imshow(strip[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]], cmap='gray', aspect = 1)
            plt.show()"""
    return words

       
# given a word, segment and return the characters within
def extract_characters(word, avg_width):
    word = 255 - word
    avg_width = int(avg_width)
    height, width = word.shape
    # use a window and find the black minima within for segmentation
   # window_x = 0
    #window_x2 = avg_width
  
    
    cutoff_points = []
    characters = []
   
    # find connected components (characters and connected characters)
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    components = []
    for i in range(n_labels):
        if(stats[i][2] >= avg_width and stats[i][2] <= avg_width * 2):
            components.append(stats[i])
    
    word = 255 - word
    
    plt.figure(figsize = (500,4))
    plt.imshow(word, cmap='gray', aspect = 1)
    plt.show()
    
    
    # try to seperate characters that are too wide, as they are possibly multiple connected characters
    connected_characters = []
    for i in range(len(components)):
        if(stats[i][2] >= 1.5*avg_width):
            connected_characters.append(i)
        # if no separation is necessary, just append the found component
        else:
            characters.append(word[:, components[i][0]:components[i][0] + components[i][2]])
           

    for component in connected_characters:
        histogram = np.zeros(width)
        # count black pixels for every column of the characer(s)
        for i in range(width):
            histogram[i] = height - np.count_nonzero(word[:, i])
        mx = max(histogram)
        mn = min(histogram)
        avg = np.mean(histogram)
        std = np.std(histogram)
        
        # if a column has too few black pixels, separate the two components
        for i in range(len(histogram)):
            if(histogram[i] <= (avg - (1.2 * std))): 
                separate = True
                cutoff_points.append(i)
        # if separation is needed, separate the component using all the found
        # cutoff points
        x1 = 0
        x2 = 0
        for point in cutoff_points:
            x2 = point
            characters.append(word[:, x1:x1 + x2])
            x1 = point
        characters.append(word[:, x1:])
        # if no separation is necessary, just append the found component
        
        
        
        
           
        
    """while(window_x2 <= width):
        window = word[0:height, window_x:window_x2]
        window_x += avg_width
        window_x2 += avg_width
        plt.figure(figsize = (500,4))
        plt.imshow(window, cmap='gray', aspect = 1)
        plt.show()
        
        
        histogram = np.zeros(width)
        # count black pixels for every column of the window
        for i in range(width):
            histogram[i] = height - np.count_nonzero(word[:, i])
        # find minima
        mx = max(histogram)
        mn = min(histogram)
        avg = np.mean(histogram)
        std = np.std(histogram)
        for i in range(len(histogram)):
            if(histogram[i] <= (avg - (2 * std))):           
                cutoff_points.append(i)
        
    cutoff_points.append(width-1)"""
    
        
   # use the cutoff points to cut the characters from the word image
    """characters = []
    x1 = cutoff_points[0]
    x2 = 0
        for i in range(len(cutoff_points)):
        x2 = cutoff_points[i]
        characters.append(word[:, x1:x2])"""
    
    for character in characters:
        if(character.shape[1] > 15):
            plt.figure(figsize = (500,4))
            plt.imshow(character, cmap='gray', aspect = 1)
            plt.show()

    
    


# given a dead sea scroll, recognize the words contained within
def recognize_handwriting(image):
    # extract the binarized parchment from the image
    parchment = extract_parchment(image, 100)
    # detect words and characters in the parchment
    boxes, centroids, avg_height, avg_width = segment_words(parchment)
    # divide parchment into line strips containing words and characters
    strips = segment_line_strips(boxes, centroids, parchment, avg_height, avg_width)
    #for each line strip, split words into characters and recognize characters
    kernel = np.ones((3,3),np.uint8)
    parchment_morphed = cv2.morphologyEx(parchment, cv2.MORPH_CLOSE, kernel)
    parchment_morphed = cv2.morphologyEx(parchment_morphed, cv2.MORPH_OPEN, kernel)
    for strip in strips:
        """plt.figure(figsize = (500,4))
        plt.imshow(parchment[strip[0]:strip[1], :], cmap='gray', aspect = 1)
        plt.show()"""
        # extract words from the strips
        words = extract_words(parchment[strip[0]:strip[1], :])    
        for word in words:
            # extract the characters in the word
            characters = extract_characters(cv2.morphologyEx(word, cv2.MORPH_CLOSE, kernel), avg_width)
            # recognize the characters in the word
            """
            for character in characters:
                recognize_characters(characters)"""
            
        
    
        