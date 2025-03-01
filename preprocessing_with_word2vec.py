# Import necessary libraries
import numpy as np
import pandas as pd
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

# Download NLTK resources
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')

# Load the IMDB dataset from Hugging Face
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
train_data = pd.DataFrame(dataset['train'])
test_data = pd.DataFrame(dataset['test'])
data = pd.concat([train_data, test_data], ignore_index=True)

# Text Preprocessing
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens  # Return tokens instead of a string for Word2Vec training

# Apply preprocessing
print("Preprocessing text data...")
data['tokens'] = data['text'].apply(preprocess_text)

# Train Word2Vec Model
print("Training Word2Vec model...")
embedding_dim = 100
word2vec_model = Word2Vec(sentences=data['tokens'], vector_size=embedding_dim, window=5, min_count=2, workers=4)

# Tokenization and Padding
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['text'].apply(lambda x: ' '.join(x)))  # Convert tokens back to text for tokenizer

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(data['text'].apply(lambda x: ' '.join(x)))

# Pad sequences
max_length = 200
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Extract labels
y = data['label'].values

# Create an embedding matrix using the Word2Vec model
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data and Word2Vec model
import pickle

with open("preprocessed_data_word2vec.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test, embedding_matrix, tokenizer), f)

word2vec_model.save("word2vec_model.bin")

print("Preprocessing complete! Data is ready for training.")
