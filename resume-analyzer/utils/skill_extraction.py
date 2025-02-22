# utils/skill_extraction.py
import spacy
import nltk
from nltk.corpus import stopwords

# Download stopwords from NLTK
nltk.download('stopwords')

# Load SpaCy model for Named Entity Recognition (NER)
nlp = spacy.load("en_core_web_sm")

# Remove stopwords from NLTK
stop_words = set(stopwords.words('english'))

def extract_skills(text):
    """Extract skills using Named Entity Recognition (NER) and filtering stopwords."""
    doc = nlp(text)
    
    # Extract skills/entities using NER
    skills = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":  # Customize to suit your NER model
            skills.append(ent.text.lower())  # Normalize skill names to lowercase
    
    # Remove stopwords and lemmatize the words
    filtered_tokens = [
        token.lemma_ for token in doc if token.text not in stop_words
    ]
    
    return list(set(skills + filtered_tokens))  # Merge skills and filtered tokens
