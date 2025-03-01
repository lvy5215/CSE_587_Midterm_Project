# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
import pickle
from sklearn.metrics import confusion_matrix, classification_report

# Load preprocessed data
with open("preprocessed_data_glove.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, embedding_matrix, tokenizer = pickle.load(f)

# Define model parameters
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_dim = 100  # Pre-trained GloVe embeddings (100D)
max_length = 200  # Fixed input sequence length
num_filters = 128  # Number of CNN filters
kernel_size = 5  # Kernel size for CNN
lstm_units = 100  # Number of LSTM units
dropout_rate = 0.5  # Dropout rate for regularization

# Build the Hybrid CNN-BiLSTM Model
input_layer = Input(shape=(max_length,))

# Embedding layer with pre-trained GloVe embeddings
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                            weights=[embedding_matrix], input_length=max_length, 
                            trainable=False)(input_layer)

# CNN Layer for feature extraction
conv_layer = Conv1D(filters=num_filters, kernel_size=kernel_size, activation='relu', padding='same')(embedding_layer)
pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)

# BiLSTM Layer for sequential learning
bilstm_layer = Bidirectional(LSTM(lstm_units, return_sequences=False))(pooling_layer)

# Fully Connected Layer
dense_layer = Dense(128, activation='relu')(bilstm_layer)
dropout_layer = Dropout(dropout_rate)(dense_layer)

# Output Layer
output_layer = Dense(1, activation='sigmoid')(dropout_layer)

# Compile the Model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# Train the Model
batch_size = 64
epochs = 3

model.fit(X_train, y_train, validation_data=(X_test, y_test), 
          batch_size=batch_size, epochs=epochs, verbose=1)

# Save the trained model
model.save("cnn_bilstm_sentiment_model.h5")

print("Model training complete and saved successfully!")

# Load the trained model for evaluation
model = load_model("cnn_bilstm_sentiment_model.h5")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Generate predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels

# Confusion matrix and classification report
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
