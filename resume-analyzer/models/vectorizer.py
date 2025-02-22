# models/vectorizer.py
import joblib
import os

def load_vectorizer():
    """Load the pre-trained vectorizer from a file."""
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.joblib')
    return joblib.load(vectorizer_path)
