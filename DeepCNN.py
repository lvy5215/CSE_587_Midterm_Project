# Import necessary libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Dense, Dropout, BatchNormalization, Add, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
import pickle

# Load preprocessed data
with open("/content/drive/MyDrive/CSE_587_Midterm_Project/preprocessed_data_glove.pkl", "rb") as f:
    X_train, X_test, y_train, y_test, embedding_matrix, tokenizer = pickle.load(f)

# Define key parameters
embedding_dim = 100  # Matches the GloVe embedding dimension
vocab_size = len(tokenizer.word_index) + 1
max_length = 200  # Matches the padded sequence length

# Define Deep CNN Model
def build_deep_cnn():
    inputs = Input(shape=(max_length,))

    # Embedding Layer with pre-trained GloVe embeddings
    x = Embedding(input_dim=vocab_size, output_dim=embedding_dim, 
                  weights=[embedding_matrix], input_length=max_length, 
                  trainable=False)(inputs)
    
    # First Conv Block
    x = Conv1D(256, kernel_size=5, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Second Conv Block
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Residual Connection
    residual = Conv1D(128, kernel_size=1, padding='same')(x)
    x = Add()([x, residual])  # Skip connection

    # Third Conv Block
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Global Pooling and Fully Connected Layers
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    # Output Layer (Binary Classification)
    outputs = Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs)
    return model

# Build and compile the model
model = build_deep_cnn()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint("best_sentiment_cnn.h5", save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=20,  # Increased epochs for better learning
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

# Evaluate on test set
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

# Save the final model
model.save("final_sentiment_analysis_cnn.h5")
print("Model saved as final_sentiment_analysis_cnn.h5")
