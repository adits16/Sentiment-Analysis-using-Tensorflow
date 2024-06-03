import pandas as pd
import numpy as np
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import joblib
import string
from spacy.lang.en.stop_words import STOP_WORDS
from flask import Flask, request, jsonify, render_template
import nltk
import tensorflow as tf

# Load trained Pipeline
model = tf.keras.models.load_model('sentiment_analysis_model.h5')


# Create the app object
app = Flask(__name__)


# creating a function for data cleaning
#from custom_tokenizer_function import CustomTokenizer
nlp = spacy.load('en_core_web_sm')
tokenizer = joblib.load('tokenizer.pkl')

# Define custom tokenizer
class CustomTokenizer:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def text_data_cleaning(self, sentence):
        doc = self.nlp(sentence)
        tokens = [token.lemma_.lower().strip() for token in doc if token.lemma_ != '-PRON-']
        tokens = [token for token in tokens if token not in nlp.Defaults.stop_words and token not in string.punctuation]
        return ' '.join(tokens)
custom_tokenizer = CustomTokenizer()

# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()][0]  # Extract the review text
    cleaned_review = custom_tokenizer.text_data_cleaning(new_review)  # Clean the review

    # Tokenize and pad the input
    seq = tokenizer.texts_to_sequences([cleaned_review])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=model.input_shape[1])

    # Make prediction
    predictions = model.predict(padded)
    sentiment = "Positive" if predictions[0] >0.9 else "Negative"
    return render_template('index.html', prediction_text=f'Sentiment: {sentiment}')



if __name__ == "__main__":
    app.run(debug=True)