# models/model.py
import joblib
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import os

def load_model():
    """Load the pre-trained model from a file."""
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.joblib')
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def train_and_save_model():
    """Train and save the model."""
    # Get the absolute path to the CSV file
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'UpdatedResumeDataSet.csv')

    # Load your dataset
    df = pd.read_csv(data_path)  # Correct file path

    # Preprocess text (you can add your preprocessing functions here)
    df['Resume'] = df['Resume'].apply(lambda x: x.lower())  # Example of simple preprocessing

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1, 2), stop_words='english', min_df=3)
    X = vectorizer.fit_transform(df['Resume'])

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['Category'])

    # Train the model
    model = LinearSVC(C=1.0, class_weight='balanced', dual=False)
    model.fit(X, y)

    # Save the model and vectorizer
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.joblib')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vectorizer.joblib')

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Model saved at {model_path}")
    print(f"Vectorizer saved at {vectorizer_path}")

if __name__ == "__main__":
    train_and_save_model()
