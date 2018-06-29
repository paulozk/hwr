import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import random
from sklearn.utils import resample

# augment the image in a variety of possible ways
def augment(image):
    kernel = np.ones((3,3),np.uint8)
    dilation = 0.1
    shear = 0.1
    # random chance of dilation
    if(random.random() <= dilation):
        image = cv2.dilate(image,kernel,iterations = 1)
        image = cv2.erode(image,kernel,iterations = 1)
         
    return image

def write_characters(character, N, source_dir, target_dir):
    word_path = os.path.join(target_dir, character)
    if(not(os.path.exists(word_path))):
        os.makedirs(word_path)
        
    #print("word:", word_path)
    character_path = os.path.join(source_dir, character + "_jpg")
    #print("ch:",character_path)
        
    for n in range(N):
        ran = random.randint(0, len(os.listdir(character_path)) - 1)
        character = os.listdir(character_path)[ran]
        path = os.path.join(character_path, character)
        
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        #image = 255 - image
        
        
        #image = cv2.resize(image, (width,height)) 
        ret1,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
        
        image =  augment(image)
      
        save_path = word_path + "\\" + str(n) 
        
        cv2.imwrite(save_path + '.jpg', image)
        
    

def create_and_write(bigram, N, source_dir, target_dir):
    
    new_bigram = []
    for character in bigram.split('_'):
        if(character == "Tasdi"):
            character = "Tsadi-medial"
        if(character == "Tsadi"):
            character = "Tsadi-medial"  
        if(character == "Tasdi-final"):
            character = "Tsadi-final"
        if(character == "Tasdi-medial"):
            character = "Tsadi-medial"
        new_bigram.append(character)
    bigram = "_".join(new_bigram)
   
        
    
    word_path = os.path.join(target_dir, bigram)
    if(not(os.path.exists(word_path))):
        os.makedirs(word_path)
    
    # flip the string for correct character sequence
    bigram = bigram.split('_')
    bigram.reverse()
    bigram = "_".join(bigram)
    print("Ngram:", bigram)
    
    kernel = np.ones((5,5),np.uint8)
    for n in range(N):
        # store the character images of a word
        ch_images = []
        # plot randomly selected characters from the second word
        for character in bigram.split('_'):
            
            # get character folder
            character_path = os.path.join(source_dir, character + "_jpg")
            ran = random.randint(0, len(os.listdir(character_path)) - 1)
            character = os.listdir(character_path)[ran]
            character_path = os.path.join(character_path, character)
            image = cv2.imread(character_path, cv2.IMREAD_GRAYSCALE)
            #image = 255 - image
            
            # remove noise from character
            #image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            _,image = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
            #image = cv2.resize(image, (width,height)) 
            #plt.imshow(image, cmap='gray', aspect = 1)
            #plt.show()
            ch_images.append(image)

        # initialize word image
        """word = np.zeros((height, width * len(ch_images)))
        word[:, 0:width] = ch_images[0]
        
        # append character pixels to word
        for i in range(len(ch_images) - 1):
            #ran = random.randint(-1, 1)
            #alignment = (24 * (i+1)) + ran
            alignment = width * (i + 1)
            word[:,alignment:alignment + width] = ch_images[i + 1]
            
        word = cv2.resize(word, (width,height)) 
        ret1,word = cv2.threshold(word,127,255,cv2.THRESH_BINARY)
        
        #word = augment(word)
        

        
        #plt.imshow(word, cmap='gray', aspect = 1)
        #plt.show()
        
        """
        # first character determines the size
        height, width = ch_images[0].shape
        for i in range(len(ch_images)):
            ch_images[i] = cv2.resize(ch_images[i], (width,height))
            
            
        #print(ch_images[0].shape)
        #print(ch_images[1].shape)
        word = np.concatenate(ch_images, axis = 1)
        
        word = augment(word)
        
        # concatenate the first two characters
        #np.concatenate((a, b), axis=0)
        save_path = word_path + "\\" + str(n) 
        #print(save_path)
        
        cv2.imwrite(save_path + '.jpg', word)
    

def generate_words():
    base_path = 'upsampled_monkbrill_aug'
    target_folder = 'dataset5'
    root_dir = os.getcwd()
    source_dir = os.path.join(root_dir, base_path)
    target_dir = os.path.join(root_dir, target_folder)
    
    
    # load the ngrams
    ngrams = pd.read_excel('../n_grams_probs.xlsx')
    #ngrams = np.asarray(ngrams)
    # filter the infrequent ngrams
    #ngrams = ngrams[:1000]

    #print("N ngrams:", len(ngrams))
    
    words = list(ngrams['Names'])
    
   
  
   
    
    # change directory to character data
    os.chdir(source_dir)
    # get all data folders
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    #print(all_subdirs)
    #plt.figure(figsize = (50,1))
    
    
    
    # amount of images generated per word
    N = 500
    folders = 0
    
    # generate character images first
    for ch in all_subdirs:
        
        write_characters(ch.split('_')[0], N, source_dir, target_dir)
        
    
    for word in words:
        print("Ngram #:", words.index(word))
        # only process bi/trigrams
        if(len(word.split("_")) < 5):
            create_and_write(word, N, source_dir, target_dir)
            folders += 1
        if(folders > 1000):
            break
    
    
    
    
    
    
    
