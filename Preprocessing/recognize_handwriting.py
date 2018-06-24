import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

from recognize_word_window import *
from markov_chain_ngram import *
from recognize_word_lstm import *

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Bidirectional, LSTM, CuDNNLSTM
from keras.optimizers import RMSprop


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
    parchment[labels == 0] = 255
    # _,parchment = cv2.threshold(parchment, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    height, width = parchment.shape
    line_image = parchment.copy()
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
            
            #cv2.line(parchment,(0, min_height),(width, min_height),(0, 200,0), 4)   
            
            strips.append([min_height, max_height])
            
    #plt.figure(figsize = (500,10))
    #plt.imshow(parchment, cmap='gray', aspect = 1)
    #plt.show()
    return strips, parchment, line_image

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
        if(stat[4] > 1000 and stat[2] < 200):
            word = strip[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]]
            words.append(word)
            """plt.figure(figsize = (500,4))
            plt.imshow(strip[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]], cmap='gray', aspect = 1)
            plt.show()"""
    return words

       
def write_line_to_file(words, path, codes):
    name = path.split('/')[1].split('.')[0]
    print(name)
    f= open("recog_output/" + name + ".txt","a+", encoding="utf-8")
    for word in words:
        if(type(word) == list):
            new_word = ""
            for ch in word:
                try:
                    code = codes[ch]
                except:
                    print("Something empty somehow ended up here..")
                    continue
                print("ch:", ch)
                print("code:", code)
                print("chr:", chr(code))
                new_word = new_word + chr(code)
            f.write(new_word + " ")
            
        else:    
            f.write(word + " ")
    f.write("\n")
    
            
def create_char_codes():
    codes = {}
    codes = {"alef": 1488, "ayin": 1506,  "bet": 1489,  "dalet": 1491,  "gimel": 1490,  "he": 1492,  "het": 1495, 
             "kaf": 1499,  "kaf-final": 1498,  "lamed": 1500,  "mem": 1501,  "mem-medial": 1502,  "nun-final": 1503, 
             "nun-medial": 1504,  "pe": 1508,  "pe-final": 1507,  "qof": 1511,  "resh": 1512,  "samekh": 1505,  "shin": 1513, 
             "taw": 1514,  "tet": 1496,  "tsadi-final": 1509,  "tsadi-medial": 1510,  "waw": 1493,  "yod": 1497,  "zayin": 1494
    }
    return codes
            
def load_model():   
    json_file = open('lstm/model.json', 'r') 
    loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    model.load_weights("lstm/stored_weights")
    return model

# given a dead sea scroll, recognize the words contained within
def recognize_handwriting(image, path, plot):
    character_codes = create_char_codes()
    # get markov transition matrix and initial probability vector
    all_prob, first_chars_prob = get_markov()
    #print("Transition matrix:", all_prob)
    #print("Initial prob vector:", first_chars_prob)
    
    # extract the binarized parchment from the image
    parchment = extract_parchment(image, 60)
    # detect words and characters in the parchment
    boxes, centroids, avg_height, avg_width = segment_words(parchment.copy())
    print("Average word height and width:", avg_height, avg_width)
    # divide parchment into line strips containing words and characters
    strips, whitened_parchment, line_image = segment_line_strips(boxes, centroids, parchment, avg_height, avg_width)
    #for each line strip, split words into characters and recognize characters
    k_size = int(avg_height / 10) 
    kernel = np.ones((k_size,k_size),np.uint8)
    parchment_morphed = cv2.morphologyEx(whitened_parchment, cv2.MORPH_CLOSE, kernel)
    #parchment_morphed = cv2.morphologyEx(parchment_morphed, cv2.MORPH_OPEN, kernel)
    
    if(plot):
        plt.figure(figsize=(500, 10))
        plt.imshow(line_image, cmap='gray', aspect=1)
        plt.show()
        plt.figure(figsize = (500,10))
        plt.imshow(parchment_morphed, cmap='gray', aspect = 1)
        plt.show()
    
    
    # load the lstm model
    model = load_model()
    print("LSTM model:", model.summary())
    for strip in strips:
        recognized_words = []
        """plt.figure(figsize = (500,4))
        plt.imshow(parchment[strip[0]:strip[1], :], cmap='gray', aspect = 1)
        plt.show()"""
        # extract words from the strips
        words = extract_words(parchment_morphed[strip[0]:strip[1], :])    
        for word in words:
            print("Recognizing the following word:")
            # extract the characters in the word
            #characters = extract_characters(cv2.morphologyEx(word, cv2.MORPH_CLOSE, kernel), avg_width)
            print("-----CNN:-------")
            recognized_word = recognize_word(cv2.morphologyEx(word, cv2.MORPH_CLOSE, kernel), avg_width,  all_prob, first_chars_prob, plot)
            #recognized_words.append(recognized_word)
            print("-----LSTM:-------")
            recognize_word_lstm(cv2.morphologyEx(word, cv2.MORPH_CLOSE, kernel), model, avg_width, plot)
        #write_line_to_file(recognized_words, path, character_codes)
             