# utils/text_processing.py
import spacy
import re

# Load SpaCy model for lemmatization
nlp = spacy.load("en_core_web_sm")

def preprocess_text(text):
    """Preprocess the text by removing non-alphanumeric characters, stopwords, and lemmatizing."""
    # Remove non-alphanumeric characters
    text = re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())).strip()

    # Use SpaCy to tokenize, lemmatize, and remove stopwords
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc if not token.is_stop])
    
    return lemmatized_text
