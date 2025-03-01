# CSE_587_Midterm_Project
In this project we did sentiment analysis on the IMDB dataset, classifying a movie review as positive or negative. We have used Glove and Word2Vec embeddings and used different deep learning models to classify the reviews.

# Running the code
First finish the data preprocessing with Glove or Word2Vec and generate pickle files.
!python preprocessing_with_glove.py
!python preprocessing_with_word2vec.py

Update the pkl file paths in the model that you want to run and run it.
!python LSTM.py

Note: 
1. All the files are ran in colab by mounting the google drive, so change paths accordingly.
2. Data is automatically downloaded from Hugging face IMDB dataset.
