from flask import Flask, render_template, request, jsonify,send_file
from flask_cors import CORS
import random
import json
import pickle
import numpy as np
import nltk
import pyttsx3
import speech_recognition as sr
from flask import Flask, jsonify, request
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import csv
from datetime import datetime
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import io
from gtts import gTTS

app = Flask(__name__)

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

CORS(app)

# Import the routes
nltk.download('wordnet')

# Register the blueprints

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/audiobot')
def audiobot():
    return render_template('audiobot.html')

@app.route('/mood_trackers')
def mood_trackers():
    return render_template('mood_trackers.html')

@app.route('/music')
def music():
    return render_template('music.html')

# Load chatbot model and related data
model = load_model('chatbot_model.keras')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()
with open('intents.json') as file:
    intents = json.load(file)
# Function to preprocess input sentence
def preprocess(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for w in sentence_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)
# Mood tracker
def record_mood(mood):
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open('mood_tracker.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([date, mood])
    return "Mood recorded successfully."

def plot_mood_data():
    dates = []
    moods = []
    with open('mood_tracker.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            dates.append(datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S'))
            moods.append(row[1])

    plt.plot(dates, moods, marker='o')
    plt.xlabel('Date')
    plt.ylabel('Mood')
    plt.title('Mood Tracker')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert plot to base64 image
    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    base64_img = base64.b64encode(img_data.getvalue()).decode()
    plt.close()

    return base64_img

@app.route('/record_mood', methods=['POST'])
def api_record_mood():
    data = request.json
    mood = data['mood']
    response = record_mood(mood)
    return jsonify({"message": response})

@app.route('/plot_mood_data', methods=['GET'])
def api_plot_mood_data():
    base64_img = plot_mood_data()
    return jsonify({"image": base64_img})


# Route to handle chatbot predictions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['message']
    processed_input = preprocess(data)
    result = model.predict(np.array([processed_input]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(result) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    response = get_response(return_list)
    return jsonify({"response": response})

# Function to get a response from intents list
def get_response(intents_list):
    tag = intents_list[0]['intent']
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

if __name__ == '__main__':
    app.run(debug=True)


