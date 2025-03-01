# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report

# Load preprocessed data
import pickle

with open("preprocessed_data_word2vec.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, embedding_matrix, tokenizer = pickle.load(f)

# Define the RNN Model(LSTM)
embedding_dim = 100  # Should match the GloVe embedding dimension
vocab_size = len(tokenizer.word_index) + 1
max_length = 200  # Should match the padded sequence length

model = Sequential()

# Embedding layer (using pre-trained GloVe embeddings)
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, weights=[embedding_matrix], input_length=max_length, trainable=False))

# Bidirectional LSTM layer
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))

# Fully connected layer
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))

# Output layer (binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the Model
# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping]
)

# Evaluate the Model
# Evaluate on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
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

# Save the Model
model.save("sentiment_analysis_rnn_model.h5")
print("Model saved as sentiment_analysis_rnn_model.h5")