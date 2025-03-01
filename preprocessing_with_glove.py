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

# Download NLTK resources (punkt and stopwords)
print("Downloading NLTK resources...")
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # Ensure punkt_tab is downloaded

# Load the IMDB dataset from Hugging Face
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")
train_data = pd.DataFrame(dataset['train'])
test_data = pd.DataFrame(dataset['test'])
data = pd.concat([train_data, test_data], ignore_index=True)

# Text Preprocessing
# Define a function to clean and preprocess text
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
    # Join tokens back into a single string
    return ' '.join(tokens)

# Apply preprocessing
print("Preprocessing text data...")
data['text'] = data['text'].apply(preprocess_text)

# Tokenization and Padding
# Initialize the tokenizer
tokenizer = Tokenizer(num_words=10000)  # Limiting vocabulary to the top 10,000 words
tokenizer.fit_on_texts(data['text'])

# Convert text to sequences of integers
sequences = tokenizer.texts_to_sequences(data['text'])

# Pad sequences to a fixed length
max_length = 200
X = pad_sequences(sequences, maxlen=max_length, padding='post', truncating='post')

# Extract labels
y = data['label'].values

# Load GloVe Embeddings
print("Loading GloVe embeddings...")
embedding_dim = 100  # Taking 100D
glove_embeddings = {}

# Loading local glove file
with open("/content/drive/MyDrive/CSE_587_Midterm_Project/data/glove.6B.100d.txt", "r", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        glove_embeddings[word] = vector

# Create an embedding matrix for the vocabulary
vocab_size = len(tokenizer.word_index) + 1
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in tokenizer.word_index.items():
    if i < vocab_size:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data for later use (optional)
import pickle

with open("preprocessed_data_glove.pkl", "wb") as f:
    pickle.dump((X_train, X_test, y_train, y_test, embedding_matrix, tokenizer), f)

print("Preprocessing complete! Data is ready for training.")