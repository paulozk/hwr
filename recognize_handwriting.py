import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola
from scipy.ndimage import rotate

from Classification.recognize_word_window import *
from Classification.recognize_word_lstm import *

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Bidirectional, LSTM, CuDNNLSTM
from keras.optimizers import RMSprop

from Preprocessing.parchment_extraction import *
from Preprocessing.word_segmentation import segment_words
from Preprocessing.line_segmentation import *


def extract_words(strip):
    kernel = np.ones((5,5),np.uint8)
    strip_morphed = 255 - strip
    
    strip_morphed = cv2.erode(strip_morphed, kernel, iterations = 1)
    strip_morphed = cv2.dilate(strip_morphed, kernel, iterations = 4)
    
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
        x = stat[0]
        y = stat[1]
        width = stat[2]
        height = stat[3]
        area = stat[4]
        
        if(area > 1000 and width < 300):
            word = strip[y:y + height, x:x + width]
            words.append(word)
            """plt.figure(figsize = (500,4))
            plt.imshow(strip[stat[1]:stat[1] + stat[3], stat[0]:stat[0] + stat[2]], cmap='gray', aspect = 1)
            plt.show()"""
    return words

       
def write_line_to_file(words, path, codes):
    f= open("test.txt","a+", encoding="utf-8")
    for word in words:
        if(type(word) == list):
            new_word = ""
            for ch in word:
                try:
                    code = codes[ch]
                except:
                    #print("Something empty somehow ended up here..")
                    continue
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
    json_file = open('Classification/lstm/best_model/model.json', 'r')
    loaded_model_json = json_file.read()
    
    model = model_from_json(loaded_model_json)
    model.load_weights("Classification/lstm/best_model/stored_weights")
    return model

# given a dead sea scroll, recognize the words contained within
def recognize_handwriting(image, path, plot):
    character_codes = create_char_codes()
    # extract the binarized parchment from the image
    print("---------------------------------------------")
    print("Extracting parchment from image..")
    parchment = extract_parchment(image, 60)
    print("Parchment extracted!")
    print()
    # detect words and characters in the parchment
    print("---------------------------------------------")
    print("Extracting word properties/boxes from binarized parchment..")
    boxes, centroids, avg_height, avg_width = segment_words(parchment.copy())
    print("Word properties extracted!")
    print()
    # divide parchment into line strips containing words and characters
    print("---------------------------------------------")
    print("Extracting lines from the parchment..")
    strips, whitened_parchment, line_image = segment_line_strips(boxes, centroids, parchment, avg_height, avg_width)
    print("Lines extracted from parchment!")
    print()
    #for each line strip, split words into characters and recognize characters
    k_size = int(avg_height / 10) 
    kernel = np.ones((k_size,k_size),np.uint8)
    parchment_morphed = cv2.morphologyEx(whitened_parchment, cv2.MORPH_CLOSE, kernel)
    
    if(plot):
        plt.figure(figsize=(500, 10))
        plt.imshow(line_image, cmap='gray', aspect=1)
        plt.show()
        plt.figure(figsize = (500,10))
        plt.imshow(parchment_morphed, cmap='gray', aspect = 1)
        plt.show()

    # load the lstm model
    model = load_model()

    for strip in strips:
        print("---------------------------------------------")
        print("Recognizing line ", str(strips.index(strip)) + ":")
        recognized_words = []
        # extract words from the strips
        words = extract_words(parchment_morphed[strip[0]:strip[1], :])    
        for word in words:
            print("Recognizing word ", str(words.index(word)) + ":")
            chars, sequence, confidences = recognize_word(word, avg_width, plot)
            final_sequence = recognize_word_lstm(word, chars, sequence, confidences, model, avg_width, plot)
            # if no characters are found, then don't add anything
            if(final_sequence != ""):
                recognized_words.append(final_sequence)
        write_line_to_file(recognized_words, path, character_codes)
        print("Recognized words have been written to output file!")
             
