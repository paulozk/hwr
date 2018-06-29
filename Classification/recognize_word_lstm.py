import os
import cv2
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_predictions(model, image, n_chars):
     # use the lstm for prediction
    output = model.predict(image)
    #print("Output:",output)
    output_sorted = np.sort(output)
    #output_sorted = output_sorted[0]
    #print("sorted output:", output_sorted)
    
    output_trimmed = output_sorted[:, 27 - n_chars:]
    
    
    mn = np.min(output_trimmed)
    #mn = 0.4
    output_labels = np.where(output >= mn)
    output_labels = output_labels[1]
    
    #print("Best indices:", output_trimmed)
    final_output = []
    
    labels = {0: "alef", 1: "ayin",  2: "bet",  3: "dalet",  4: "gimel",  5: "he",  6: "het", 
                     7: "kaf",  8: "kaf-final",  9: "lamed",  10: "mem",  11: "mem-medial",  12: "nun-final", 
                     13: "nun-medial",  14: "pe",  15: "pe-final",  16: "qof",  17: "resh",  18: "samekh",  19: "shin", 
                     20: "taw",  21: "tet",  22: "tsadi-final",  23: "tsadi-medial",  24: "waw",  25: "yod",  26: "zayin"
    }
    
  
    
    
    print("Idx:", output_labels)
    for idx in output_labels:
        final_output.append(labels[idx])
 
        
    return final_output, output_trimmed.tolist()
    
    
# apply domain knowledge: medial characters have to be in the middle of the word,
# final characters have to be the final character in the word
# If any of them are in the wrong spot, the confidence in those predictions should
# be set to zero
def filter_with_knowledge(labels, confidences): 
    
    # check if a -final character is anywhere but in the final position
    for i in range(len(labels) - 1):
        if(len(labels[i].split('-')) > 1 and labels[i].split('-')[1] == "final"):
            confidences[i] = 0
        
    
    
    # first check if label can be split in two
    if(len(labels[0].split('-')) > 1):
        # check if first character in the sequence has 'final' or 'medial' in the name, which is not allowed
        if(labels[0].split('-')[1] == 'medial'):
            confidences[0] = 0
    if(len(labels[-1].split('-')) > 1):
        # check if final character in predicted sequence has medial in the name, which is also not allowed
        if(labels[-1].split('-')[1] == "medial"):
            confidences[-1] = 0
    
    return labels, confidences
    
def illegal_character(labels, index):
    if(len(labels[index].split('-')) > 1):
        if(labels[index].split('-')[1] == "final" and index < (len(labels) -1)):
            return True
        elif(labels[index].split('-')[1] == "medial" and (index == 0 or (len(labels) -1))):
            return True
    return False
            
    
    
def classify(word_preds, word_probs, char_preds, char_probs, cnn_sequence, cnn_confidences):
    
    # filter the confidences of the labels by using domain knowledge  (final and medial characters)
    char_preds, char_probs = filter_with_knowledge(char_preds, char_probs)
    cnn_sequence, cnn_confidences = filter_with_knowledge(cnn_sequence, cnn_confidences)
    
    
   
    
    similar_characters = {
        "bet": ["kaf", "mem-medial", "pe"],
        "kaf": ["bet", "mem-medial", "pe"],
        "mem-medial": ["kaf", "bet", "pe"],
        "pe": ["kaf", "Mem-medial", "bet"],
        
        "alef": ["ayin"],
        "ayin": ["alef"],
        
        "he": ["het, taw"],
        "het": ["he, taw"],
        "taw": ["he", "het"],
        
        "dalet": ["resh", "kaf-final", "nun-final", "pe-final"],
        "resh": ["dalet", "kaf-final", "nun-final", "pe-final"],
        "kaf-final": ["resh", "dalet", "nun-final", "pe-final"],
        "nun-final": ["resh", "kaf-final", "dalet", "pe-final"],
        "pe-final": ["resh", "kaf-final", "nun-final", "dalet"],
        
        #"tsadi-medial" : ["tsadi-final"],
        #"tsadi-final" : ["tsadi-medial"],
        
        "waw": ["yod"],
        "yod": ["waw"]
        
    }
   
    print("Word preds:", word_preds)
    print("Word probs:", word_probs)
    print("ch preds:", char_preds)
    print("ch probs:", char_probs)
    
    n_chars = len(word_probs)
    
    final_output = {}
    final_sequence = [""] * n_chars
    
    
    positions = []
    for n in range(n_chars):
        positions.append(n)
    
    
    # first try to find matching characters in word and character predictions
    for i in range(n_chars):
        matching_chars = []
        word_pred = word_preds[i]
        # 1: is word prediction in character predictions?
        for j in range(len(char_preds)):
            char_pred = char_preds[j]
            if(word_pred == char_pred):
                if(char_probs[j] >= 0.5):
                    final_output[j] = char_preds[j]
                
                
                
        # 2: is a similar character from the word prediction in character predictions?
        if(word_pred in similar_characters.keys()):
            similar_chars = similar_characters[word_pred]
            for j in range(len(char_preds)):
                for similar_ch in similar_chars:
                    # check if character at position j matches a similar character
                    if(char_preds[j] == similar_ch):
                        # check if position is already filled
                        if(j not in final_output.keys()):
                            # check if the word prediction or the word prediction
                            # is more confident
                            ch_conf = char_probs[j]
                            word_conf = word_probs[i]
                            cnn_conf = cnn_confidences[j]
                                                      
                            # LSTM is more confident in its character prediction compared to the others
                            if(ch_conf > word_conf and ch_conf > cnn_conf):
                                final_output[j] = char_preds[j]                   
                            # LSTM is more confident in its word prediction compared to the others
                            elif(word_conf > ch_conf and word_conf > cnn_conf):
                                final_output[j] = word_preds[i]
                            # CNN is more confident than LSTM 
                            else:
                                final_output[j] = cnn_sequence[j]
                                
                                
    # if all positions still have not been filled, use the cnn classifications for 
    # those positions!
    for p in positions:
        # position in sequence still empty?
        if(p not in final_output.keys()):
            # fill the position with the CNN prediction or word prediction, depending on which is higher, unless character is not allowed there (conf == 0).
            print("Empty position:", p)
            print("LSTM: ",  char_preds[p] , char_probs[p])
            print("CNN: ",  cnn_sequence[p] , cnn_confidences[p])
            if(cnn_confidences[p] != 0 or char_probs[p] != 0):
                if(char_probs[p] > cnn_confidences[p]):
                    final_output[p] = char_preds[p]
                else:
                    final_output[p] = cnn_sequence[p]
            # else, look at the LSTM word prediction and take the character not yet in the sequence with the highest
            # confidence
            else:
                # store characters that are not in the sequence yet and are not illegal (final / medial)
                characters_left = []
                probs = []
                for i in range(len(word_preds)):
                    if(word_preds[i] not in list(final_output.values()) and not (illegal_character(word_preds, i))):
                        characters_left.append(i)
                        probs.append(word_probs[i])
                # get the character with the highest LSTM word classification confidence
                best = max(probs)
                best_idx = probs.index(best)
                final_output[p] = word_preds[best_idx]
                
                        
                    
                        
                
                
        
    # use the dictionary to build the sequence
    for pos in final_output.keys():
        final_sequence[pos] = final_output[pos]
                            
    print("Final output:", final_sequence)
    
    return final_sequence
                    
                
                
    
    
    


# recognize the word using the trained LSTM model 
def recognize_word_lstm(word, chars, cnn_sequence, cnn_confidences, model, avg_width, plot):

    n_chars = len(chars)
    
    print("N chars:", n_chars)
    
    if(n_chars > 0):
    
        kernel = np.ones((3,3))

        # invert the colors
        word = 255 - word
        print("orig shape:",word.shape)

        _,word = cv2.threshold(word,127,255,cv2.THRESH_BINARY)

        #word = cv2.dilate(word, kernel, iterations = 1)
        #word = cv2.erode(word, kernel, iterations = 1)


        # show the word
        if(plot):
            """im = cv2.imread('4.jpg', 0)

            plt.figure(figsize = (500,4))
            plt.imshow(im, cmap='gray', aspect = 1)
            plt.show()"""

            plt.figure(figsize = (500,4))
            plt.imshow(word, cmap='gray', aspect = 1)
            plt.show()


        word = cv2.resize(word, (32, 32))
        word = word / 255
        word = np.reshape(word, (1,32,32))

        word_preds, word_probs = get_predictions(model, word, n_chars)
        word_probs = word_probs[0]

        char_preds = []
        char_probs = []
        for ch in chars:
            plt.imshow(ch, cmap='gray', aspect = 1)
            plt.show()
            ch = cv2.resize(ch, (32, 32))
            ch = ch / 255
            ch = np.reshape(ch, (1,32,32))
            preds, probs = get_predictions(model, ch, 1)
            char_preds.append(preds[0])
            char_probs.append(probs[0][0])


        # combine the word and character perspectives to output a final classification
        return classify(word_preds, word_probs, char_preds, char_probs, cnn_sequence, cnn_confidences)
    return ""
        
        
    
    
    
    
