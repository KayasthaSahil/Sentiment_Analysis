# Imports Flask for web serving, pickle for loading the model, os for file paths, re for regex, and scikit-learn/nltk for text processing.
from flask import Flask, render_template, request, session
import pickle
import os
import re
from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud
import io
import base64
from markupsafe import Markup, escape
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Ensure VADER lexicon is available at server start
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

#Defines a custom text cleaning transformer for use in a scikit-learn pipeline.
#Removes HTML, non-word characters, and extracts emojis.
#Pipeline Integration: Inheriting from BaseEstimator and TransformerMixin allows this to be used in scikit-learn pipelines, making training and inference consistent.
class TextCleaner(BaseEstimator, TransformerMixin):
    def remove_html(self, text):
        return re.sub('<[^>]*>', '', text)

    def remove_non_words(self, text):
        return re.sub('[\W]+', ' ', text.lower())

    def extract_emojis(self, text):
        emojis = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
        return ' '.join(emojis).replace('-', '')

    def preprocess(self, text):
        text = self.remove_html(text)
        text = self.remove_non_words(text)
        emojis = self.extract_emojis(text)
        return text + ' ' + emojis

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Accept both pandas Series and lists
        if hasattr(X, 'apply'):
            return X.apply(self.preprocess)
        else:
            return [self.preprocess(x) for x in X]

nltk.download('stopwords', quiet=True)
stop = stopwords.words('english')
porter = PorterStemmer()
#Defines a tokenizer that removes stopwords and applies stemming.
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split() if word not in stop]

def generate_wordcloud_base64(text, stopwords_list=None, width=400, height=200):
    """
    Generate a word cloud image from text, encode as base64, and return the string.
    Returns None if not enough words.
    """
    if not text or len(text.split()) < 3:
        return None
    stopwords_set = set(stopwords_list) if stopwords_list else set()
    wc = WordCloud(
        width=width,
        height=height,
        background_color='white',
        stopwords=stopwords_set,
        collocations=False
    )
    wc.generate(text)
    img_io = io.BytesIO()
    wc.to_image().save(img_io, format='PNG')
    img_io.seek(0)
    img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
    return img_base64

class SentimentModel:
    """Handles loading and prediction for the sentiment analysis model."""
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    def predict(self, text):
        return self.model.predict([text])[0]

    def predict_with_confidence(self, text):
        label = self.model.predict([text])[0]
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba([text])[0]
            if hasattr(self.model, 'classes_'):
                idx = list(self.model.classes_).index(label)
            else:
                idx = 1 if label == 1 else 0
            confidence = proba[idx]
        else:
            confidence = 1.0
        return label, round(confidence * 100, 2)

def highlight_sentiment_words(text):
    sia = SentimentIntensityAnalyzer()
    import re
    tokens = re.findall(r"\w+|[\s.,!?;]", text)
    highlighted = []
    for token in tokens:
        word = token.strip().lower()
        if not word or not word.isalpha():
            highlighted.append(escape(token))
            continue
        score = sia.polarity_scores(word)['compound']
        if score >= 0.5:
            highlighted.append(f'<span class="positive-word">{escape(token)}</span>')
        elif score <= -0.5:
            highlighted.append(f'<span class="negative-word">{escape(token)}</span>')
        else:
            highlighted.append(escape(token))
    return Markup(''.join(highlighted))

from functools import wraps
PREDICTION_HISTORY_KEY = 'prediction_history'
MAX_HISTORY = 5

def add_prediction_to_history(user_text, label, confidence):
    try:
        confidence_py = float(confidence)
    except Exception:
        confidence_py = confidence
    entry = {
        'input': user_text,
        'label': str(label),
        'confidence': confidence_py
    }
    history = session.get(PREDICTION_HISTORY_KEY, [])
    history.append(entry)
    history = history[-MAX_HISTORY:]
    session[PREDICTION_HISTORY_KEY] = history
    session.modified = True
    return history

def get_prediction_history():
    return session.get(PREDICTION_HISTORY_KEY, [])

class SentimentApp:
    def __init__(self, model_path):
        self.app = Flask(__name__)
        self.app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_secret_key')
        self.model = SentimentModel(model_path)
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/', methods=['GET', 'POST'])
        def index():
            prediction = None
            confidence = None
            user_text = ''
            wordcloud_img = None
            wordcloud_fallback = None
            highlighted_text = None
            prediction_history = get_prediction_history()
            if request.method == 'POST':
                user_text = request.form.get('user_text', '')
                if user_text:
                    prediction, confidence = self.model.predict_with_confidence(user_text)
                    cleaner = TextCleaner()
                    cleaned = cleaner.preprocess(user_text)
                    cleaned_words = [w for w in cleaned.split() if w not in stop]
                    cleaned_text = ' '.join(cleaned_words)
                    wordcloud_img = generate_wordcloud_base64(cleaned_text, stop)
                    if not wordcloud_img:
                        wordcloud_fallback = 'Not enough words to generate a word cloud.'
                    highlighted_text = highlight_sentiment_words(user_text)
                    prediction_history = add_prediction_to_history(user_text, prediction, confidence)
            return render_template('index.html', prediction=prediction, confidence=confidence, user_text=user_text, wordcloud_img=wordcloud_img, wordcloud_fallback=wordcloud_fallback, highlighted_text=highlighted_text, prediction_history=prediction_history)

model_path = os.path.join(os.path.dirname(__file__), 'sentiment_model.pkl')
app_instance = SentimentApp(model_path)
handler = app_instance.app 