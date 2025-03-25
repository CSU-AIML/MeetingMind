import os
import sys

def install_dependencies():
    os.system("pip install -r requirements.txt")
    
    # Download NLTK data
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('names', quiet=True)
    
    # Download spaCy model
    os.system("python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    install_dependencies()
