# pdf_extraction.py
import PyPDF2

def extract_text_from_pdf(pdf_path):
    text = []
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = [page.extract_text() for page in reader.pages]
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return " ".join(text).lower()
