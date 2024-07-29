from flask import Flask, render_template, request
import pickle
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import os

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

app = Flask(__name__, static_folder='static')


def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [stemmer.stem(word) for word in text if word not in stop_words]
        text = ' '.join(text)
        return text
    else:
        return ''


@app.route('/')
def index():
   return render_template('index.html', prediction='')

@app.route('/predict', methods=['POST'])
def predict_news():
    News_Article = str(request.form.get('News_Article'))
    cleaned_text = preprocess_text(News_Article)

    news_article_numeric = vectorizer.transform([cleaned_text]).toarray()
    result = model.predict_news(news_article_numeric)

    if result[0] == 0:
        prediction = "Fake"
    else:
        prediction = "Real"

    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    model_path = 'logistic_model.pkl'
    model = pickle.load(open(model_path, 'rb'))

    vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 3))
    df_fake = pd.read_csv("Fake.csv")
    df_true = pd.read_csv("True.csv")
    df_fake["class"] = 0
    df_true["class"] = 1
    df = pd.concat([df_fake, df_true], axis=0)
    df = df.sample(frac=1)
    corpus = df['title'].tolist()
    vectorizer.fit(corpus)

    app.run(debug=True)