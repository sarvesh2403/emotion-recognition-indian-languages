import pandas as pd
import numpy as np
import tensorflow as tf
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional, Input, Attention, Layer
)
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Download stopwords
nltk.download('stopwords')

# Initialize stopwords and stemmer
english_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\u0900-\u097F\u0C80-\u0CFF\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    words = [ps.stem(word) if word not in english_stopwords else '' for word in text.split()]
    return ' '.join(words)

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def call(self, inputs):
        query, value = inputs
        scores = tf.matmul(query, value, transpose_b=True)  # Compute attention scores
        weights = tf.nn.softmax(scores, axis=-1)  # Normalize to get attention weights
        output = tf.matmul(weights, value)  # Compute weighted output
        return output

# Load datasets
hindi_data = pd.read_csv('/content/drive/MyDrive/hindi_emotion_dataset.csv')
english_data = pd.read_csv('/content/drive/MyDrive/english_emotion_dataset.csv')
kannada_data = pd.read_csv('/content/drive/MyDrive/kannada_emotion_dataset.csv')

# Clean and combine datasets
hindi_data.columns = ['text', 'Emotion']
english_data.columns = ['text', 'Emotion']
kannada_data.columns = ['text', 'Emotion']
hindi_data['text'] = hindi_data['text'].astype(str).apply(clean_text)
english_data['text'] = english_data['text'].astype(str).apply(clean_text)
kannada_data['text'] = kannada_data['text'].astype(str).apply(clean_text)
combined_data = pd.concat([hindi_data, english_data, kannada_data], ignore_index=True).sample(frac=1, random_state=42)

# Preprocess text and labels
texts = combined_data['text'].astype(str).values
emotions = combined_data['Emotion'].values
label_encoder = LabelEncoder()
encoded_emotions = label_encoder.fit_transform(emotions)

# Tokenization
max_vocab_size = 10000
max_length = 50
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_emotions, test_size=0.2, random_state=42)

# Define the model with Custom Attention
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(input_dim=max_vocab_size, output_dim=128, input_length=max_length)(input_layer)
dropout_layer = SpatialDropout1D(0.2)(embedding_layer)
bilstm_layer = Bidirectional(LSTM(64, return_sequences=True))(dropout_layer)

# Attention Mechanism
query = bilstm_layer
value = bilstm_layer
attention_output = AttentionLayer()([query, value])

# Flatten and Dense layers
dense_layer = Dense(64, activation='relu')(attention_output[:, -1, :])  # Extract only the last timestep
output_layer = Dense(len(np.unique(encoded_emotions)), activation='softmax')(dense_layer)

# Build and compile the model
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=64,
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Predict on test data
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)  # Get the class with the highest probability for each prediction

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Visualize confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Save the model and tokenizer
model.save('multilingual_emotion_model_with_attention.h5')
import pickle
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle)

# Prediction function
def predict_emotion(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded = pad_sequences(sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    return predicted_label

# Test predictions
test_text = "मैं आज बहुत खुश हूं!"  # Hindi
print(f"Predicted Emotion: {predict_emotion(test_text)}")
test_text_2 = "ನಾನು ಇಂದು ಸಂತೋಷವಾಗಿದ್ದೇನೆ!"  # Kannada
print(f"Predicted Emotion: {predict_emotion(test_text_2)}")
