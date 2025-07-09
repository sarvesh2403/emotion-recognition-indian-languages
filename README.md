# Emotion Recognition in Multilingual Text Using BiLSTM-Based Approach for Indian Languages

This project implements a deep learning-based emotion classification system that works across **multiple Indian languages**, including **English, Hindi, and Kannada**. It leverages a **Bidirectional LSTM (BiLSTM)** architecture enhanced with a **custom attention mechanism** for better contextual understanding and emotion recognition from text.

---

## 📌 Overview

- **Goal**: Classify text into emotion categories such as *joy, sadness, anger, fear, surprise*, etc.
- **Languages Supported**: English, Hindi, Kannada
- **Architecture**: BiLSTM with a custom attention layer
- **Dataset**: English Twitter dataset (~10 lakh entries reduced to 1 lakh), manually balanced and translated to Hindi and Kannada using Excel
- **Framework**: TensorFlow / Keras

---

## 🧠 Model Architecture

```
Input Text
   ↓
Embedding Layer
   ↓
SpatialDropout1D
   ↓
Bidirectional LSTM
   ↓
Custom Attention Layer
   ↓
Dense Layer → Softmax Output
```

---

## 📁 Project Structure

```
emotion-recognition-indian-languages/
│
├── data/
│   ├── english_emotion_dataset.csv
│   ├── hindi_emotion_dataset.csv
│   └── kannada_emotion_dataset.csv
│
├── models/
│   ├── multilingual_emotion_model_with_attention.h5
│   └── tokenizer.pkl
│
├── src/
│   └── train_multilingual_emotion_model.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sarvesh2403/emotion-recognition-indian-languages.git
cd emotion-recognition-indian-languages
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python src/train_multilingual_emotion_model.py
```

This will:
- Load and preprocess multilingual data
- Train a BiLSTM model with attention
- Save the model and tokenizer

---

## 🔍 Emotion Prediction Example

```python
from src.train_multilingual_emotion_model import predict_emotion

print(predict_emotion("मैं आज बहुत खुश हूं!"))         # Hindi
print(predict_emotion("ನಾನು ಇಂದು ಸಂತೋಷವಾಗಿದ್ದೇನೆ!"))  # Kannada
print(predict_emotion("I'm feeling so sad today."))     # English
```

---

## 📊 Evaluation

- Accuracy and Confusion Matrix used for evaluation
- Visualized using seaborn heatmap
- Final test accuracy printed at the end of training

---

## 📚 Dataset Notes

- Source: English Twitter dataset
- Size: Originally ~10 lakh samples, reduced to 1 lakh
- Balancing: Manually balanced for class distribution
- Multilingual Support: Translated to Hindi and Kannada using Excel

---

## 🔧 Technologies Used

- Python
- TensorFlow / Keras
- NLTK
- Scikit-learn
- Matplotlib
- Seaborn

---
## 👥 Team Members

- **Sarvesh Hampali**  
  GitHub: [@sarvesh2403](https://github.com/sarvesh2403)

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---
