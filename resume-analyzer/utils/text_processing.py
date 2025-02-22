# text_processing.py
import re

CLEAN_PATTERN = re.compile(r'[^a-zA-Z0-9\s]')
WS_PATTERN = re.compile(r'\s+')

def preprocess_text(text):
    return WS_PATTERN.sub(' ', CLEAN_PATTERN.sub(' ', text.lower())).strip()
