from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Initialize FastAPI app
app = FastAPI()

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

# Load the dataset and label column
data = pd.read_csv('fake.csv')
label = np.random.randint(0, 2, len(data))
data['label_col'] = label

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def load_and_preprocess_data(data):
    df = data
    df['text'] = df['text'].apply(preprocess_text)
    return df

def train_model(df):
    X = df['text']
    y = df['label_col']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train_tfidf, y_train)

    y_pred = pac.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')

    conf_matrix = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)

    return tfidf_vectorizer, pac

def predict_fake_news(text, tfidf_vectorizer, model):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_tfidf)[0]
    return "Potentially Fake" if prediction == 0 else "Likely Reliable"

# Load and preprocess the data
df = load_and_preprocess_data(data)
tfidf_vectorizer, model = train_model(df)

# Request Body model for input
class NewsItem(BaseModel):
    text: str

# FastAPI prediction endpoint
@app.post("/predict")
def predict(news_item: NewsItem):
    prediction = predict_fake_news(news_item.text, tfidf_vectorizer, model)
    return {"prediction": prediction}
