import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

nltk.download('punkt')

def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def tokenize(text):
    return nltk.word_tokenize(text)

def preprocess(text):
    return tokenize(clean_text(text))

def tfidf_vectorizer(corpus):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def train_word2vec(sentences):
    tokenized = [preprocess(s) for s in sentences]
    model = Word2Vec(tokenized, vector_size=100, window=5, min_count=1)
    return model
