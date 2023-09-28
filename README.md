# Word-Search-Solver
Description: Takes an image of any given word search, scans the pixel data and then uses a naïve Bayesian classifier to estimate all the letters, then using a given word list, finds these words among the given letters that the artificial intelligence generated

Features: 
reduce_dimensions - Using Principal Component Analysis (PCA) to reduce the dimensions of any given data set to N_Dimensions.

process_training_data - Processes the labelled data to return a model stored as a dictionary with the correct formatting

classify_squares - Takes a feature vector and a model and then will classify and label the features with a 92% accuracy. 

find_words - Goes through the labels given by classify_squares to check for any words in a given list, checking all directions and always finding all appropriate words.
