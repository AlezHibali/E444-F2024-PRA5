from flask import Flask, jsonify, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# Initialize the Flask app
application = Flask(__name__)

# Load the ML model and vectorizer
def load_model():
    # Load the classifier
    with open('basic_classifier.pkl', 'rb') as fid:
        loaded_model = pickle.load(fid)

    # Load the vectorizer
    with open('count_vectorizer.pkl', 'rb') as vd:
        vectorizer = pickle.load(vd)

    return loaded_model, vectorizer

model, vectorizer = load_model()

# Route for fake news prediction
@application.route('/predict', methods=['POST'])
def predict():
    # Get the news article from the request body
    data = request.json
    news_article = data.get('article')

    if not news_article:
        return jsonify({'error': 'No article provided'}), 400

    # Transform the input data using the vectorizer
    transformed_data = vectorizer.transform([news_article])

    # Predict using the loaded model
    prediction = model.predict(transformed_data)[0]

    # Return the prediction result ('FAKE' or 'REAL')
    return jsonify({'prediction': prediction})

# Launch app
if __name__ == "__main__":
    application.run()

# curl -X POST http://ece-444-pra5-env-2.eba-ujsdbkn2.us-east-1.elasticbeanstalk.com/predict -H "Content-Type: application/json" -d "{\"article\": \"This is fake news\"}" 
