import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from Classification.recognize_character_cnn import *
import itertools
import pandas as pd

    
def water_drop(component, avg_width, plot):
    # store distance fallen at every column of the component
    distances = []
    cutoff_points = []
    
    good_images = []
    good_chars = []
    confidences = [] 
    
    first_part = 0.2 * component.shape[1]
    last_part = 0.8 * component.shape[1]
    
    for i in range(component.shape[1]):
        # start at the top of the current column
        drop = 0
        # ignore the first and last part of the image
        if(i > first_part and i < last_part):  
            # let the drop fall until it hits a part of the character (white pixel)
            while(component[drop,i] == 0):
                #component[drop,i] = 125
                drop += 1
        distances.append(drop)
            
    # plot the result
    #if(plot):
        #plt.figure(figsize = (500,4))
        #plt.imshow(component, cmap='gray', aspect = 1)
        #plt.show()   
        
    
    
    # get the drop that fell the highest
    best = max(distances)
    best_idx = distances.index(best)
    cutoff_points.append(best_idx)
    
    rng = 10
    gap = avg_width / 1.5
    latest_point = 0
    # find other drop locations that are within the range of the lowest
    # flalen drop
    for i in range(len(distances)):
        if(distances[i] != best_idx and (best - rng) <= distances[i] and ((i - latest_point) >= gap or (i == 0))):
            latest_point = i
            cutoff_points.append(i)
            
    start = 0
    end = 0
    
    
    for i in range(len(cutoff_points)):
        end = cutoff_points[i]
        # filter out weird super-thin pieces
        if(component[:, start:end].shape[1] > 5): 
            good_images.append(component[:, start:end])
        start = end
        
    
    
    #plt.figure(figsize = (500,4))
    #plt.imshow(component[:, start:], cmap='gray', aspect = 1)
    #plt.show() 
    good_images.append(component[:, start:])
    
    # classify the separated characters
    for ch in good_images:
        print(ch.shape)
        plt.figure(figsize = (500,4))
        plt.imshow(ch, cmap='gray', aspect = 1)
        plt.show()
        top_chars, top_scores = recognize_character(ch)
        best_score = max(top_scores)
        best_idx = top_scores.index(best_score)
        best_char = top_chars[best_idx].split()[0]
        print("Best score:", best_score)
        print("Best character:", best_char) 
        good_chars.append(best_char)
        confidences.append(best_score)
        
    good_images.reverse()
    good_chars.reverse()
    confidences.reverse()
    

    
    return good_images, good_chars, confidences
   

# recognize the characters in a word by sliding a window across big
# connected components and recognizing characters within
def recognize_word(word, avg_width, plot):
    
    word = 255 - word
    # binarize the word, just in case
    _,word = cv2.threshold(word,127,255,cv2.THRESH_BINARY)
    
 
 
    # show the word
    if(plot):
        plt.figure(figsize = (500,4))
        plt.imshow(word, cmap='gray', aspect = 1)
        plt.show()
        
    # find the connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(word, 8, cv2.CV_32S)
    # remove background component
    stats = stats[1:] 
    stats = stats[stats[:,0].argsort()]
    # flip the characters for the Hebrew right-to-left reading
    stats = np.flip(stats, 0)
    
    # store the entire predicted sequence
    sequence = []
    # store the confidence per character
    confidences = []
    # store the character images
    chars = []
    
    for stat in stats:  
        # characters' height or width should be high enough 
        if(stat[2] > (0.4 * avg_width) or (stat[3] > (0.5 * word.shape[0]))):  
            ch = word[stat[1]: stat[1]+ word.shape[0], stat[0]: stat[0]+stat[2]]
            print("Component:")
            plt.figure(figsize = (500,4))
            plt.imshow(ch, cmap='gray', aspect = 1)
            plt.show()
            
            # components that are too big need to be processed further, as they probably consist
            # of multiple characters;
            # apply the sliding window technique to the long component to find the characters within
            if(stat[2] > avg_width * 1.1):
                #good_images, good_chars, confidences = slide_window(ch, avg_width, plot) 
                good_images, good_chars, confs = water_drop(ch, avg_width, plot)
                #water_drop(ch, avg_width, plot)
              
                # put words into main sequence
                for i in range(len(good_images)):
                    chars.append(good_images[i])
                    sequence.append(good_chars[i])
                    confidences.append(confs[i])
                
            # else just classify the component
            else:
                chars.append(ch)
                top_chars, top_scores = recognize_character(ch)
                best_score = max(top_scores)
                best_idx = top_scores.index(best_score)
                best_char = top_chars[best_idx].split()[0]
                print("Best score:", best_score)
                print("Best character:", best_char)
                sequence.append(best_char)
                confidences.append(best_score)    
                
    #chars, sequence, confidences = slide_window(word, avg_width, plot)
    print("Predicted sequence:", sequence, confidences)
    return chars, sequence, confidences
   

    
    
    