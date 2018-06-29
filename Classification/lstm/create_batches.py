import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import random



def construct_label(word):
    labels = {"alef": 0, "ayin": 1,  "bet": 2,  "dalet": 3,  "gimel": 4,  "he": 5,  "het": 6, 
             "kaf": 7,  "kaf-final": 8,  "lamed": 9,  "mem": 10,  "mem-medial": 11,  "nun-final": 12, 
             "nun-medial": 13,  "pe": 14,  "pe-final": 15,  "qof": 16,  "resh": 17,  "samekh": 18,  "shin": 19, 
             "taw": 20,  "tet": 21,  "tsadi-final": 22,  "tsadi-medial": 23,  "waw": 24,  "yod": 25,  "zayin": 26
    }
    # construct label based on characters in word
    label = [0] * 27
    for ch in word.split("_"):
        ch = ch.lower()
        index = labels[ch]
        label[index] = 1
    
    return label
        

def create_batches(path, n, size):
    print(os.listdir(path))
    
    batches = []
    labels = []
    
    
    
    # add data from all word folders
    for word in os.listdir(path):
      
        #if(os.listdir(path).index(word) > 0):
            #break
        # construct label based on type of word/ngram
        label = construct_label(word)
        
        # add each image to the batch
        i = 0
        for image in os.listdir(path + "\\" + word):
            im_path = path + "\\" + word + "\\" + image
            im = cv2.imread(im_path, 0)
            im = cv2.resize(im, (size,size))
            im = im / 255
            
            batches.append(im)
            labels.append(label)
            i += 1
            if(i == n):
                break
        print("Batch ", word, "done.")
        
    
    labels = np.asarray(labels)
    
    print("Labels shape:", labels.shape)
    
    print(len(batches))
    
    batches = np.asarray(batches)
    #batches = np.reshape(batches, (len(batches), timesteps, data_dim))
      
    
    #batches = np.reshape(batches, (len(batches), 150, 9000))
    print("Batches shape:", batches.shape)
    rng_state = np.random.get_state()
    
    np.random.shuffle(batches)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
   
    
    return batches, labels
    
    
    
    
    
    
    
    
