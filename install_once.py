import os
import subprocess
import pickle

# Path to marker file that indicates packages are installed
MARKER_FILE = 'packages_installed.flag'

def install_packages():
    print("Installing required packages (one-time setup)...")
    packages = [
        "jax==0.4.20",
        "transformers==4.30.2",
        "torch==2.0.1",
        "streamlit==1.22.0",
        "openai-whisper",
        "nltk==3.8.1",
        "spacy==3.5.3",
        "scikit-learn==1.2.2",
        "librosa==0.10.1",
        "openai",
        "fpdf==1.7.2",
        "reportlab==4.0.4",
        "markdown==3.4.3",
        "beautifulsoup4==4.12.2",
        "jellyfish==0.11.2",
        "python-Levenshtein==0.21.1"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.check_call(["pip", "install", package])
    
    # Download required NLTK data
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('names', quiet=True)
    
    # Install spaCy model
    subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
    # Create marker file to indicate packages are installed
    with open(MARKER_FILE, 'w') as f:
        f.write('Packages installed successfully')
    
    print("All packages installed successfully!")

# Check if packages are already installed
if not os.path.exists(MARKER_FILE):
    install_packages()
else:
    print("Packages already installed. Skipping installation.")

if __name__ == "__main__":
    # Force installation if run directly
    if os.path.exists(MARKER_FILE):
        os.remove(MARKER_FILE)
    install_packages()
