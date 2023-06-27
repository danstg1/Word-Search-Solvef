"""
Naive Bayes Classifier used to extract and find words from a word search
By Daniel St-Gallay for COM2004
Supported Code DCS Sheffield
"""

from typing import List
import numpy as np
from utils import utils
from utils.utils import Puzzle
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.linalg
from difflib import SequenceMatcher


def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 50
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.
    This takes in the raw data, and a model if it exists , and produces a reduced training model
    If the model does not exist, then use the data itself as training data.
    Uses PCA Reduction
    """
    try:
        train_data = np.array(model['fvectors_train']) #If the model hasn't been created yet , use the test data as the train data
    except:
        train_data = data

    test_data = data

    covariance = np.cov(train_data, rowvar=0)

    N = covariance.shape[0] # Get the Vector Length

    w, v = scipy.linalg.eigh(covariance, eigvals=(N - N_DIMENSIONS, N - 1))
    v = np.fliplr(v)
    reduced_data = np.dot((test_data - np.mean(train_data)), v)

    return reduced_data


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.
    Formats the data correctly, so that it has the letter (label) followed by the data it holds
    , then uses baesian classifcation to create a model that can be used and check against
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    model["fvectors_train"] = fvectors_train.tolist()


    data_points = []
    testdataLst = []
    pset = []

    for j in range(25): #Per Letter In The Alphabet
        letter_lst = []
        
        for i in range(len(model['labels_train'])): # Go through ever data point

            if (model['labels_train'][i] == chr(j+65)): # If that data point is the current letter, add this data to this specific letter's list
                letter_lst.append(utils.flatten([[ord(model['labels_train'][i])-64], model['fvectors_train'][i]])) # With the letter converted as a number
        

        label_data= np.array(letter_lst)

        train_data = label_data[0::2, :]

        test_data = label_data[1::2, :]

        
        mean1 = np.mean(train_data.astype(np.float), axis=0)
        cov1 = np.cov(train_data[:, 1:], rowvar=0)
        dist1 = multivariate_normal(mean=mean1[1:], cov=np.diag(cov1))

        data_points.append(dist1)

        testdataLst.append(test_data)



    
    #TRAINING STAGE COMPLETE

    return model

def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """
    Uses baesian classifcation to take i nthe fvectors and the model, to then 
    attempt to correctly label every letter in the system
    """
    
    model['fvectors_train'] = reduce_dimensions(np.array(model['fvectors_train']), model)
    pset = []
    data_points = []
    for j in range(26): #Per Letter In The Alphabet
        letter_lst = []
        
        for i in range(len(model['labels_train'])): # Go through ever data point

            if (model['labels_train'][i] == chr(j+65)): # If that data point is the current letter, add this data to this specific letter's list
                letter_lst.append(utils.flatten([[ord(model['labels_train'][i])-64], model['fvectors_train'][i]])) # With the letter converted as a number
        

        letetr_data= np.array(letter_lst)

        train_data = letetr_data[0::2, :]
        
        mean1 = np.mean(train_data.astype(np.float), axis=0)
        cov1 = np.cov(train_data[:, 1:], rowvar=0)
        
        dist1 = multivariate_normal(mean=mean1[1:], cov=np.diag(cov1))

        data_points.append(dist1) #Same as above, find the MVN for all letters


    for i in range(len(data_points)):
        pset.append(data_points[i].pdf(fvectors_test))

    p = np.vstack(tuple(pset))
    index2 = np.argmax(p, axis=0)
    char_lst = []
    
    for i in range(len(index2)):
        char_lst.append(chr(int(index2[i]+1)+64))

    return char_lst

def test_vectors(labels,current_row, current_col, direction):
    """
    A function to test if the next cords are valid or if they would go off the word search, e.g cords [0,-1] would fail
    """
    if current_row + direction[0] >= labels.shape[0]:
        return False
    
    elif current_col + direction[1] >= labels.shape[1]:
        return False

    elif (current_row + direction[0] < 0 ) or (current_col + direction[1] < 0 ):
        return False

    else:
        return True

def check_surroundings(next_letter, current_col, current_row, labels):
    """
    Takes in next letter you are trying to find, the current row and col you are on and outputs the position of those letters.
    """
    cords = []
    directions = [[0,-1],[1,-1],[1,0],[1,1],[0,1],[-1,1],[-1,-1],[-1,0]]
    
    for direction in directions:

        if (test_vectors(labels,current_row,current_col, direction)):
            if next_letter == labels[current_row+direction[0]][current_col+direction[1]]:
                cords.append(direction)


    return cords

d = ""    
    
def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """
    Runs through every word in the list, for each word it attempts to find
    2 letters next to each other that match the start of the word,
    it then continues to check the letters in that direction to see if 
    the word is complete.
    If the word is not complete, it returns the closest to a complete word
    that it got
    """

    word_pos = []

    for i in range(len(words)): #For Every Word To Be Found
    
        
        current_word = words[i].upper()
        best_attempt = ""
        best_cords = (0,0,len(current_word),0)

        
        for j in range(labels.shape[0]): #Check Every Row
            
            for k in range(labels.shape[1]): #Check Every Col

                if labels[j][k] == current_word[0]: # If the Letter at those cords are the same as the start of the word      


                    start_row = j
                    start_col = k

                    next_letter_pos = 1

                    directions = check_surroundings(current_word[next_letter_pos], start_col, start_row, labels) #Checks the surrounding letters, to see if they match 2nd letter, returns list of directions 
                    
                    next_letter_pos+=1


                    for direction in directions: #For each direction it has returned
                        current_attempt = current_word[0] + current_word[1]


                        start_row = j + direction[0] # Resets all values for new word
                        start_col = k + direction[1]
                        next_letter_pos = 2



                        if (test_vectors(labels,start_row,start_col, direction)): # If these will break the system (Go off the word Search, Dont)
                        
                            while next_letter_pos < len(current_word) and  test_vectors(labels,start_row,start_col, direction): #While the letters are still lining up and no errors are thrown


                            
                                next_letter_pos += 1
                                start_row = start_row+direction[0] # Increase the letter position and add the directions
                                start_col=start_col+direction[1]

                               


                                current_attempt = current_attempt + labels[start_row][start_col] #Increases the attempt number


                                if similar(current_word , current_attempt) > similar(current_word, best_attempt): # If this is the best attempt so far
                                    best_attempt = current_attempt
                                    best_cords = (j, k, start_row, start_col) #Store these cordinates
                                


        word_pos.append(best_cords) #Appends the best cordinates found for that word
    
                                
    

    return word_pos
