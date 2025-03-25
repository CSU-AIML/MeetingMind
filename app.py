import streamlit as st
# Set page config as the first Streamlit command
st.set_page_config(page_title="MeetingMind", page_icon="🧠", layout="wide")

# Run installation script before importing other modules
import os
import sys
import subprocess

# Check if packages are installed and install if needed
try:
    import whisper
except ImportError:
    # Run the installation script
    subprocess.call([sys.executable, "install_once.py"])

# Import all required modules
import os
import whisper
import torch
import re
import nltk
import spacy
import gc
import openai
import librosa
from datetime import timedelta
from pathlib import Path
# ...rest of your imports...
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import defaultdict
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import base64
from fpdf import FPDF
import textwrap
import markdown
from bs4 import BeautifulSoup
import io
import unicodedata
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
# import fuzzy  # For Double Metaphone algorithm
import jellyfish  # For additional phonetic algorithms and string similarity
import Levenshtein  # For edit distance calculation
from difflib import SequenceMatcher  # For string similarity
warnings.filterwarnings('ignore')
try:
    # Try importing fuzzy directly first
    import fuzzy
except ImportError:
    try:
        # Try using the wrapper if direct import fails
        from fuzzy_wrapper import fuzzy
        print("Using fuzzy wrapper")
    except ImportError:
        # Last resort fallback
        print("Warning: fuzzy package not available. Using simplified name matching.")
        # Create simple fallback implementation
        class DummyDMetaphone:
            def __call__(self, text):
                if not text:
                    return [None]
                code = text.lower()[:4]
                return [code.encode('utf-8')]
        
        class FuzzyModule:
            def __init__(self):
                self.DMetaphone = DummyDMetaphone
        
        fuzzy = FuzzyModule()
# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('words', quiet=True)
nltk.download('names', quiet=True)
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
def import_transformers_pipeline():
    """
    Attempt to import pipeline with comprehensive error handling
    """
    try:
        # First, try standard import
        from transformers import pipeline
        return pipeline
    except ImportError:
        # If standard import fails, provide detailed diagnostic information
        print("Standard Transformers pipeline import failed.")
        
        try:
            # Check Transformers library installation
            import transformers
            print(f"Transformers version: {transformers.__version__}")
            
            # List available modules
            print("Available modules:", dir(transformers))
        except ImportError:
            print("Transformers library not found at all.")
        
        # List alternative import strategies
        alternative_imports = [
            "from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline",
            "from transformers.pipelines import pipeline",
            "from transformers.pipelines.base import pipeline"
        ]
        
        for alt_import in alternative_imports:
            try:
                exec(alt_import)
                print(f"Successfully imported using: {alt_import}")
                return eval('pipeline')
            except Exception as e:
                print(f"Alternative import failed: {alt_import}")
                print(f"Error: {e}")
        
        # If all imports fail, raise a comprehensive error
        raise ImportError("""
        Unable to import pipeline from Transformers. 
        Possible solutions:
        1. Reinstall transformers: pip install transformers
        2. Check Python and transformers versions
        3. Verify library compatibility
        4. Install with: pip install 'transformers[torch]'
        """)

# Dynamically import pipeline
pipeline = import_transformers_pipeline()
def install_ffmpeg():
    """Install FFmpeg if not already present"""
    try:
        # Try to run ffmpeg to check if it's installed
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg is already installed.")
        return
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg not found. Attempting to install...")
        
        # Determine the package manager and install command
        if sys.platform.startswith('linux'):
            # For Linux (including Ubuntu/Debian)
            try:
                subprocess.run(['sudo', 'apt-get', 'update'], check=True)
                subprocess.run(['sudo', 'apt-get', 'install', '-y', 'ffmpeg'], check=True)
                print("FFmpeg installed successfully on Linux.")
            except subprocess.CalledProcessError:
                print("Failed to install FFmpeg using apt-get. You may need to install it manually.")
        
        elif sys.platform == 'darwin':
            # For macOS
            try:
                subprocess.run(['brew', 'install', 'ffmpeg'], check=True)
                print("FFmpeg installed successfully on macOS.")
            except subprocess.CalledProcessError:
                print("Failed to install FFmpeg using Homebrew. You may need to install it manually.")
        
        elif sys.platform == 'win32':
            # For Windows
            print("Please install FFmpeg manually for Windows.")
        
        else:
            print(f"Unsupported platform {sys.platform}. Please install FFmpeg manually.")

# Call the installation function
install_ffmpeg()
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants

# Replace these lines in your app.py
BASE_URL = st.secrets.get("BASE_URL", "https://models.inference.ai.azure.com")
MODEL = st.secrets.get("MODEL", "gpt-4o")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")  # Will be set in Lightning AI
# Replace with your actual token
# Add this in your app creation function
if not GITHUB_TOKEN:
    st.warning("⚠️ API key is not set. Some features may not work properly.")
class NameStandardizer:
    """Class for standardizing human names using phonetic algorithms with improved filtering"""
    def __init__(self):
        self.name_mapping = {}  # Maps variants to standardized names
        self.name_groups = {}   # Groups similar names by phonetic code
        self.metaphone_mapping = {}  # Maps metaphone codes to standard names
        self.person_context = {} # Stores context about people (roles, etc.)

        # Initialize spaCy
        self.nlp = spacy.load("en_core_web_sm")

        # Load common human names datasets
        self.load_name_datasets()

        # Create an extended set of English words (non-names)
        self.english_words = set(nltk.corpus.words.words())
        self.stopwords = set(stopwords.words('english'))

        # Common words that are often capitalized but not names
        self.common_capitalized = {
            "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
            "January", "February", "March", "April", "May", "June", "July", "August",
            "September", "October", "November", "December", "The", "A", "An", "And", "But",
            "Or", "For", "Nor", "So", "Yet", "I", "Internet", "World", "Web", "Email", "Yes",
            "No", "Ok", "Okay", "Hi", "Hello", "Thanks", "Thank", "You", "Please", "University",
            "College", "School", "Hospital", "Bank", "Station", "Airport", "Government",
            "Company", "Corporation", "Department", "Association", "Organization"
        }

    def load_name_datasets(self):
        """Load common first and last name datasets including multicultural names"""
        # Get male and female names from NLTK (primarily Western names)
        male_names = set(nltk.corpus.names.words('male.txt'))
        female_names = set(nltk.corpus.names.words('female.txt'))

        # Add common Indian names
        indian_names = {
            "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Reyansh", "Ayaan", "Atharva",
            "Krishna", "Ishaan", "Shaurya", "Advik", "Rudra", "Kabir", "Anik", "Armaan",
            "Dhruv", "Yuvan", "Virat", "Rohan", "Veer", "Gaurav", "Aahan", "Arnav",
            "Aanya", "Aadhya", "Aaradhya", "Ananya", "Pari", "Anika", "Navya", "Diya",
            "Avani", "Anaya", "Sara", "Myra", "Saanvi", "Ira", "Disha", "Kiara",
            "Riya", "Aahana", "Anvi", "Prisha", "Siya", "Divya", "Avni", "Mahika",
            "Sharma", "Patel", "Singh", "Kumar", "Gupta", "Kaur", "Shah", "Mehta",
            "Chopra", "Reddy", "Bose", "Verma", "Kapoor", "Anand", "Khanna", "Chowdhury",
            "Mukherjee", "Chatterjee", "Agarwal", "Mukhopadhyay", "Das", "Banerjee", "Desai", "Malhotra"
        }

        # Add common Muslim names
        muslim_names = {
            "Mohammed", "Muhammad", "Ahmad", "Mahmoud", "Abdullah", "Ali", "Omar", "Hassan",
            "Hussein", "Ibrahim", "Khalid", "Tariq", "Waleed", "Yusuf", "Zaid", "Bilal",
            "Hamza", "Mustafa", "Samir", "Rayan", "Malik", "Jamal", "Karim", "Amir",
            "Fatima", "Aisha", "Maryam", "Zainab", "Layla", "Noor", "Hana", "Samira",
            "Amina", "Leila", "Salma", "Yasmin", "Zahra", "Rania", "Sara", "Lina",
            "Noura", "Khadija", "Sana", "Farah", "Iman", "Nadia", "Aliyah", "Dalia",
            "Khan", "Rahman", "Ahmed", "Pasha", "Malik", "Syed", "Qureshi", "Aziz",
            "Hussain", "Farooq", "Javed", "Iqbal", "Abbas", "Rashid", "Mirza", "Asif"
        }

        # Combine all name sets
        self.first_names = male_names.union(female_names).union(indian_names).union(muslim_names)

        # Add some common name prefixes and suffixes for various cultures
        self.name_prefixes = {
            "Al", "El", "Abd", "Abdul", "Bin", "Ibn", "Abu",
            "Mc", "Mac", "De", "Van", "Von", "O'", "St.", "San"
        }

        self.name_suffixes = {
            "Jr", "Jr.", "Sr", "Sr.", "II", "III", "IV",
            "Khan", "Lal", "Devi", "Wala", "Bhai", "Ji", "Babu"
        }

        # We'll use a probability threshold for accepting names
        self.name_probability = {}
        for name in self.first_names:
            self.name_probability[name] = 1.0
            # Also add lowercase version with slightly lower probability
            self.name_probability[name.lower()] = 0.9

    def is_likely_human_name(self, word):
        """Determine if a word is likely to be a human name with multicultural awareness"""
        # If word is in our known name list, it's highly likely
        if word in self.first_names:
            return True

        # Check for name prefixes and suffixes that indicate a name
        for prefix in self.name_prefixes:
            if word.startswith(prefix) and len(word) > len(prefix) + 1:
                return True

        for suffix in self.name_suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 1:
                return True

        # Check if it's a common non-name word
        if word.lower() in self.stopwords:
            return False

        if word in self.common_capitalized:
            return False

        # Cultural name patterns
        # Check for common Indian and Muslim name patterns

        # Common Indian name endings
        indian_suffixes = ['ji', 'nath', 'raj', 'dev', 'pal', 'kar', 'wati', 'deep', 'preet', 'jeet']
        for suffix in indian_suffixes:
            if word.lower().endswith(suffix) and len(word) > len(suffix) + 2:
                return True

        # Common Muslim name patterns
        muslim_prefixes = ['al-', 'abdul', 'ibn', 'abu', 'bin', 'mohammad', 'muhammad']
        for prefix in muslim_prefixes:
            if word.lower().startswith(prefix) and len(word) > len(prefix) + 2:
                return True

        # For multicultural names, be less strict about excluding based on English dictionary
        # Only exclude if it's a very common English word AND short (as common short words are
        # unlikely to be names across cultures)
        if len(word) <= 4 and word.lower() in self.english_words and word.lower() not in self.name_probability:
            return False

        # Additional checks
        # Names typically don't have numbers
        if any(c.isdigit() for c in word):
            return False

        # Multicultural names may have special characters, but generally alphanumeric with some exceptions
        allowed_chars = "-'., "
        if any(c not in allowed_chars and not c.isalpha() for c in word):
            return False

        # Most names have a reasonable length
        if len(word) < 2 or len(word) > 30:  # Increased max length for compound names
            return False

        # This is a heuristic - names typically don't end in certain suffixes
        # But be careful with multicultural names that might use these patterns
        non_name_suffixes = ['ing', 'ed', 'ly', 'tion', 'ment', 'ness', 'ity', 'ism']
        if (word.lower().endswith(tuple(non_name_suffixes)) and
            len(word) > 5 and  # Only apply to longer words
            not any(word.lower().endswith(cultural_suffix) for cultural_suffix in indian_suffixes)):
            return False

        # If the word passed all filters and is capitalized, it might be a name
        # For multicultural names, capitalization is important but not definitive
        if word[0].isupper():
            return True

        # For non-capitalized words, check if they match cultural patterns
        # This helps with names that might be written in lowercase
        return any(word.lower().startswith(p) for p in muslim_prefixes) or any(word.lower().endswith(s) for s in indian_suffixes)

    def verify_person_entity(self, ent_text):
        """Verify a spaCy PERSON entity is likely a real name with multicultural awareness"""
        parts = ent_text.split()

        # Single-word name check
        if len(parts) == 1:
            return self.is_likely_human_name(parts[0])

        # For multi-word names, check if at least one part is in our name dataset
        for part in parts:
            if part in self.first_names:
                return True

        # Handle compound names with hyphens or special characters
        # Example: "Abdul-Rahman" or "Mohammed Al-Fayed"
        compound_parts = []
        for part in parts:
            compound_parts.extend([p for p in re.split(r'[-\s]', part) if p])

        for part in compound_parts:
            if part in self.first_names:
                return True

        # Check for cultural name patterns
        # Handle common Indian name patterns (e.g., first name + father's name or place name)
        if len(parts) >= 2:
            # Check if any part matches name patterns
            indian_suffixes = ['ji', 'nath', 'raj', 'dev', 'pal', 'kar', 'wati', 'deep', 'preet', 'jeet']
            muslim_prefixes = ['al-', 'abdul', 'ibn', 'abu', 'bin', 'mohammad', 'muhammad']

            for part in parts:
                # Check for Indian name patterns
                if any(part.lower().endswith(suffix) for suffix in indian_suffixes):
                    return True

                # Check for Muslim name patterns
                if any(part.lower().startswith(prefix) for prefix in muslim_prefixes):
                    return True

        # If no parts match known patterns, check general name patterns
        # For multicultural names, be more lenient
        if len(parts) == 2 or len(parts) == 3:
            # Check if all parts are capitalized (basic name pattern across cultures)
            if all(p[0].isupper() for p in parts):
                # Check that parts don't look like obvious verbs or common nouns
                verb_suffixes = ['ing', 'ed']
                if all(not any(p.lower().endswith(s) for s in verb_suffixes) for p in parts):
                    return True

        # For 3+ part names (common in some cultures), check capitalization
        if len(parts) >= 3:
            if all(p[0].isupper() for p in parts):
                return True

        # Default to spaCy's judgment with moderate confidence for multicultural names
        # SpaCy has been trained on diverse datasets and may recognize patterns we don't explicitly check
        return len(parts) >= 2 and all(p[0].isupper() for p in parts)

    def add_name(self, name, confidence=1.0):
        """Add a name to be processed, with confidence filter"""
        # Skip names with low confidence or very short names
        if confidence < 0.5 or not name or len(name) < 2:
            return

        # Check if this looks like a name
        if not any(part in self.first_names for part in name.split()):
            # If no part of the name is in our known names list,
            # require higher confidence
            if confidence < 0.8:
                return

        # Get the metaphone code
        try:
            metaphone_code = fuzzy.DMetaphone()(name)[0]
            if not metaphone_code:
                return

            # Convert to string for consistency
            metaphone_code = metaphone_code.decode('utf-8') if isinstance(metaphone_code, bytes) else metaphone_code

            # If this is a new metaphone code
            if metaphone_code not in self.metaphone_mapping:
                self.metaphone_mapping[metaphone_code] = name
                self.name_groups[metaphone_code] = {name}
            else:
                # Use Levenshtein distance to decide if this is truly the same name
                standard_name = self.metaphone_mapping[metaphone_code]
                distance = Levenshtein.distance(name.lower(), standard_name.lower())

                # If very similar, add to existing group
                if distance <= min(len(name), len(standard_name)) * 0.4:  # 40% threshold
                    self.name_groups[metaphone_code].add(name)

                    # Use the shorter name as standard if it's not just an initial
                    if len(name) < len(standard_name) and len(name) > 2:
                        self.metaphone_mapping[metaphone_code] = name

                    # If current name is longer and has capital letters (likely more formal)
                    elif len(name) > len(standard_name) and sum(1 for c in name if c.isupper()) > sum(1 for c in standard_name if c.isupper()):
                        self.metaphone_mapping[metaphone_code] = name

                # If this is a different name with same metaphone code
                else:
                    # Generate a unique key by appending a number
                    new_key = f"{metaphone_code}_{len([k for k in self.metaphone_mapping if k.startswith(metaphone_code)])}"
                    self.metaphone_mapping[new_key] = name
                    self.name_groups[new_key] = {name}

            # Map this variant to the standard name
            standard_name = self.metaphone_mapping[metaphone_code]
            self.name_mapping[name] = standard_name

        except Exception as e:
            print(f"Error processing name '{name}': {e}")

    def extract_names_from_text(self, text):
        """Extract potential names from text with improved filtering"""
        # Process with spaCy to find named entities
        doc = self.nlp(text)

        # Store context information
        name_indicators = ["mr", "mr.", "ms", "ms.", "mrs", "mrs.", "dr", "dr.",
                          "professor", "prof", "prof.", "sir", "madam", "miss"]

        # Dictionary to store names with their confidence scores
        extracted_names = {}

        # Extract names from entities with verification
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Check if this is likely a real name
                if self.verify_person_entity(ent.text):
                    # Check if there are name indicators before this entity
                    confidence = 0.9  # Base confidence for spaCy PERSON entities

                    # Look for name indicators in the previous tokens
                    for token in doc:
                        if token.text.lower() in name_indicators and token.i < ent.start and token.i >= ent.start - 3:
                            confidence = 1.0  # Highest confidence with name indicators

                    # Add the full name with high confidence
                    if len(ent.text.split()) > 1:
                        extracted_names[ent.text] = confidence

                    # Add individual parts with slightly lower confidence
                    for name_part in ent.text.split():
                        if name_part[0].isupper() and len(name_part) > 1:
                            # Only add individual parts that are likely names
                            if self.is_likely_human_name(name_part):
                                extracted_names[name_part] = confidence - 0.1

        # Additional pattern-based name extraction for cases spaCy might miss
        words = word_tokenize(text)
        for i, word in enumerate(words):
            # Skip words already processed
            if word in extracted_names:
                continue

            # Check for name indicators followed by capitalized words
            if i > 0 and words[i-1].lower().rstrip('.') in [ind.rstrip('.') for ind in name_indicators]:
                if word[0].isupper() and self.is_likely_human_name(word):
                    extracted_names[word] = 0.95

                    # Try to get the full name if it's a multi-word name
                    if i < len(words) - 1 and words[i+1][0].isupper() and self.is_likely_human_name(words[i+1]):
                        full_name = f"{word} {words[i+1]}"
                        extracted_names[full_name] = 0.98

        # Add all extracted names with their confidence values
        for name, confidence in extracted_names.items():
            self.add_name(name, confidence)

    def standardize_name(self, name):
        """Return the standardized version of a name"""
        if not name or len(name) < 2:
            return name

        # First verify this is likely a name
        if not self.is_likely_human_name(name.split()[0]):
            return name  # Return unchanged if not likely a name

        # If we've seen this name before, return its standard form
        if name in self.name_mapping:
            return self.name_mapping[name]

        # Try to match it by metaphone
        try:
            metaphone_code = fuzzy.DMetaphone()(name)[0]
            if metaphone_code:
                metaphone_code = metaphone_code.decode('utf-8') if isinstance(metaphone_code, bytes) else metaphone_code
                if metaphone_code in self.metaphone_mapping:
                    standard_name = self.metaphone_mapping[metaphone_code]
                    self.name_mapping[name] = standard_name
                    return standard_name
        except:
            pass

        # If no match found, add it as a new name and return itself
        if self.is_likely_human_name(name.split()[0]):
            self.add_name(name, 0.8)  # Add with moderate confidence
        return name

    def standardize_names_in_text(self, text):
        """Find and standardize all names in a text"""
        # First extract potential names
        self.extract_names_from_text(text)

        # Sort names by length (longest first) to handle substring matches properly
        names_by_length = sorted(self.name_mapping.keys(), key=len, reverse=True)

        # Replace all name occurrences
        for name in names_by_length:
            # Create a regex pattern that handles word boundaries
            pattern = r'\b' + re.escape(name) + r'\b'
            text = re.sub(pattern, self.name_mapping[name], text)

        return text

    def get_name_variants(self):
        """Return all name variants and their standardized forms"""
        return self.name_mapping

    def get_name_groups(self):
        """Return groups of names that were matched together"""
        return self.name_groups


class AudioProcessor:
    def __init__(self, model_size="small"):
        with st.spinner('Loading Whisper model...'):
            try:
                self.model = whisper.load_model(model_size)
            except Exception as e:
                st.error(f"Error loading Whisper model: {str(e)}")
                st.info("Using base model instead")
                self.model = whisper.load_model("base")

    def get_audio_duration(self, audio_path: str) -> float:
        """Get the duration of an audio file in seconds"""
        try:
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception as e:
            st.warning(f"Could not determine audio duration: {e}")
            return 0

    def format_duration(self, duration_seconds: float) -> str:
        """Format duration in seconds to HH:MM:SS format"""
        td = timedelta(seconds=int(duration_seconds))
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours} hour{'s' if hours > 1 else ''}, {minutes} minute{'s' if minutes > 1 else ''}"
        else:
            return f"{minutes} minute{'s' if minutes > 1 else ''}, {seconds} second{'s' if seconds > 1 else ''}"

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper"""
        try:
            result = self.model.transcribe(audio_path, task="translate", verbose=False)  # This will translate to English
            return result["text"]
        finally:
            # Clear GPU memory after transcription
            torch.cuda.empty_cache()
            gc.collect()


@dataclass
class PreprocessingConfig:
    remove_timestamps: bool = True
    remove_fillers: bool = True
    fix_contractions: bool = True
    remove_duplicates: bool = True
    fix_punctuation: bool = True
    segment_sentences: bool = True
    remove_stopwords: bool = False
    chunk_size: int = 512
    chunk_overlap: int = 50


class TextPreprocessor:
    def __init__(self, config: PreprocessingConfig = PreprocessingConfig()):
        self.config = config
        self.stop_words = set(stopwords.words('english'))
        self.name_standardizer = NameStandardizer()

        self.fillers = [
            "um", "uh", "like", "you know", "i mean",
            "so", "basically", "actually", "literally",
            "sort of", "kind of", "well"
        ]

        self.contractions = {
            "won't": "will not", "can't": "cannot", "n't": " not",
            "i'm": "i am", "i've": "i have", "you're": "you are",
            "you've": "you have", "we're": "we are", "we've": "we have",
            "they're": "they are", "they've": "they have",
            "it's": "it is", "that's": "that is", "what's": "what is",
            "let's": "let us", "who's": "who is"
        }

        # Compile regex patterns
        self.timestamp_pattern = re.compile(r'\[\d{2}:\d{2}:\d{2}\]')
        self.filler_pattern = re.compile(r'\b(?:' + '|'.join(self.fillers) + r')\b', re.IGNORECASE)
        self.duplicate_pattern = re.compile(r'\b(\w+)(?:\s+\1)+\b', re.IGNORECASE)
        self.punctuation_pattern = re.compile(r'([.,!?])([^\s])')
        self.spaces_pattern = re.compile(r'\s+')
        self.quotes_pattern = re.compile(r'"\s*([^"]*?)\s*"')

    def preprocess(self, text: str) -> str:
        """Apply all preprocessing steps based on configuration"""
        if self.config.remove_timestamps:
            text = self.timestamp_pattern.sub('', text)
        if self.config.remove_fillers:
            text = self.filler_pattern.sub('', text)
        if self.config.fix_contractions:
            for contraction, expansion in self.contractions.items():
                text = re.sub(rf'\b{contraction}\b', expansion, text, flags=re.IGNORECASE)
        if self.config.remove_duplicates:
            text = self.duplicate_pattern.sub(r'\1', text)
        if self.config.fix_punctuation:
            text = self.punctuation_pattern.sub(r'\1 \2', text)
            text = self.spaces_pattern.sub(' ', text)
            text = self.quotes_pattern.sub(r'"\1"', text)
            text = text.strip()

        # Extract and standardize names before segmenting
        self.name_standardizer.extract_names_from_text(text)
        text = self.name_standardizer.standardize_names_in_text(text)

        if self.config.segment_sentences:
            text = ' '.join(sent_tokenize(text))
        if self.config.remove_stopwords:
            words = text.split()
            text = ' '.join(word for word in words if word.lower() not in self.stop_words)
        return text

    def split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
            chunk = ' '.join(words[i:i + self.config.chunk_size])
            chunks.append(chunk)

        return chunks

    def get_name_standardization_report(self):
        """Get a report of name standardization"""
        name_groups = self.name_standardizer.get_name_groups()
        report = []

        for metaphone, names in name_groups.items():
            if len(names) > 1:  # Only show groups with variants
                standard_name = self.name_standardizer.metaphone_mapping.get(metaphone, list(names)[0])
                variants = [name for name in names if name != standard_name]
                if variants:  # Only report if there are actual variants
                    report.append(f"Standardized '{standard_name}' from variants: {', '.join(variants)}")

        return report


class SummaryPostProcessor:
    def __init__(self):
        with st.spinner('Loading language model...'):
            self.nlp = spacy.load("en_core_web_sm")

    def fix_sentence_boundaries(self, text: str) -> str:
        """Fix sentence boundaries and capitalization"""
        doc = self.nlp(text)
        sentences = []

        for sent in doc.sents:
            fixed_sent = sent.text.strip()
            if fixed_sent:
                fixed_sent = fixed_sent[0].upper() + fixed_sent[1:]
                if not fixed_sent[-1] in {'.', '!', '?'}:
                    fixed_sent += '.'
                sentences.append(fixed_sent)

        return ' '.join(sentences)

    def remove_incomplete_sentences(self, text: str) -> str:
        """Remove fragments and incomplete sentences"""
        doc = self.nlp(text)
        complete_sentences = []

        for sent in doc.sents:
            has_subject = False
            has_verb = False

            for token in sent:
                if token.dep_ in {'nsubj', 'nsubjpass'}:
                    has_subject = True
                if token.pos_ == 'VERB':
                    has_verb = True

            if has_subject and has_verb:
                complete_sentences.append(sent.text.strip())

        return ' '.join(complete_sentences)

    def fix_conjunctions(self, text: str) -> str:
        """Fix common conjunction issues and improve flow"""
        text = re.sub(r'\band\s+and\b', 'and', text)
        text = re.sub(r'\bbut\s+but\b', 'but', text)
        text = re.sub(r'\s+but\s+', ', but ', text)
        text = re.sub(r'\s+however\s+', ', however, ', text)
        return text

    def fix_list_formatting(self, sentences: List[str]) -> List[str]:
        """Improve formatting of list items"""
        formatted_sentences = []
        for sent in sentences:
            sent = re.sub(r'\s+•\s+', '. ', sent)
            sent = re.sub(r'^\s*•\s*', '', sent)
            sent = re.sub(r'^\d+\.\s*([a-z])', lambda m: f"{m.group(1).upper()}", sent)
            formatted_sentences.append(sent)
        return formatted_sentences

    def postprocess_summary(self, text: str) -> str:
        """Apply all post-processing steps"""
        text = self.fix_sentence_boundaries(text)
        text = self.remove_incomplete_sentences(text)
        text = self.fix_conjunctions(text)
        sentences = [s.strip() for s in re.split('[.!?]', text) if s.strip()]
        formatted_sentences = self.fix_list_formatting(sentences)
        return '. '.join(formatted_sentences) + '.'


class TextSummarizer:
    def __init__(self):
        with st.spinner('Loading BART summarization model...'):
            self.summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1,
            )
            self.nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])
            self.post_processor = SummaryPostProcessor()

    def process_chunks(self, chunks: List[str], progress_bar) -> Dict[str, List[str]]:
        """Process chunks and group by topics with post-processing"""
        grouped_chunks = defaultdict(list)
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            summary = self.summarizer(chunk, max_length=130, min_length=30)[0]['summary_text']
            processed_summary = self.post_processor.postprocess_summary(summary)
            topic = self.identify_topics(chunk)
            grouped_chunks[topic].append(processed_summary)

            # Update progress
            progress = (i + 1) / total_chunks
            progress_bar.progress(progress)

            if len(grouped_chunks) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        return grouped_chunks

    def identify_topics(self, text: str) -> str:
        """Identify the main topic of a text segment"""
        doc = self.nlp(text)

        topic_keywords = {
    # 🚀 Business & Corporate
    'Company Overview': ['company', 'organization', 'mission', 'vision', 'values'],
    'Business Strategy & Management': ['strategy', 'leadership', 'management', 'growth', 'operations', 'business model', 'entrepreneurship', 'stakeholders'],
    'Finance & Investment': ['finance', 'investment', 'funding', 'profit', 'loss', 'budget', 'capital', 'valuation', 'stocks', 'trading'],
    'Marketing & Branding': ['marketing', 'branding', 'advertising', 'SEO', 'social media', 'promotion', 'public relations'],
    'Sales & Customer Relationship': [
        'sales', 'CRM', 'lead generation', 'customer service', 'retention', 'pricing strategy',
        'follow-ups', 'quotations', 'lead status', 'client engagement', 'stakeholder meeting',
        'customer pipeline', 'business development'
    ],
    'E-Commerce & Retail': ['e-commerce', 'retail', 'supply chain', 'inventory', 'logistics', 'warehousing'],
    'Corporate Strategy & Financials': ['investors', 'corporate', 'supply chain', 'mergers', 'acquisitions'],
    'Market Expansion & Globalization': ['market', 'expansion', 'international trade', 'emerging markets'],
    'Client Engagement & Sales Strategy': [
        'client', 'POC', 'demo', 'pipeline', 'licensing', 'collateral', 'roadmap', 'proposal',
        'investment roadmap', 'business negotiations', 'partner meetings'
    ],
    'Project & Report Management': [
        'report format', 'summary report', 'documentation', 'findings', 'meeting notes',
        'proposal customization', 'contract discussions'
    ],

    # 💻 Technology & AI
    'Artificial Intelligence & Automation': [
        'AI', 'machine learning', 'deep learning', 'automation', 'data analytics', 'neural networks',
        'ML model', 'data-driven insights', 'intelligent automation', 'predictive analytics'
    ],
    'Cybersecurity & Risk Management': [
        'cybersecurity', 'HTTPS', 'encryption', 'TLS', 'attack', 'authorization', 'security',
        'penetration testing', 'ransomware', 'risk register', 'remediation', 'audit', 'SOC',
        'security compliance', 'risk-based approach', 'security audits', 'VAPT'
    ],
    'Software Development & IT': [
        'software', 'coding', 'cloud computing', 'DevOps', 'databases', 'full stack development',
        'API', 'staging', 'production', 'server', 'software customization', 'IT infrastructure'
    ],
    'Testing & Evaluation': ['testing', 'scoring', 'quantitative', 'qualitative', 'user testing', 'penetration testing'],
    'Blockchain & Cryptocurrency': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'NFTs', 'decentralization'],
    'Internet & Digital Trends': ['internet', 'IoT', 'web development', '5G', 'cloud services', 'big data'],

    # 🔬 Science & Engineering
    'Healthcare & Medicine': [
        'healthcare', 'medicine', 'hospital', 'surgery', 'doctor', 'medical research',
        'healthcare compliance', 'HIPAA', 'patient data security', 'healthcare AI solutions'
    ],
    'Biotechnology & Pharmaceuticals': ['biotech', 'pharmaceuticals', 'drug discovery', 'genomics', 'clinical trials'],
    'Engineering & Innovation': ['engineering', 'mechanical', 'electrical', 'civil', 'robotics', 'nanotechnology'],
    'Environmental Science & Sustainability': ['sustainability', 'climate change', 'carbon footprint', 'renewable energy'],
    'Energy & Emissions': ['energy', 'renewable', 'non-renewable', 'solar', 'wind', 'tidal', 'carbon emissions', 'power plants'],
    'Space Exploration & Astronomy': ['astronomy', 'NASA', 'Mars', 'black holes', 'cosmology', 'satellites'],
    'Physics & Chemistry': ['quantum mechanics', 'thermodynamics', 'chemical reactions', 'material science'],

    # 🎓 Education & Learning
    'Academic Research & Education': [
        'education', 'learning', 'teaching', 'university', 'school', 'curriculum',
        'educational technology', 'e-learning platforms'
    ],
    'Online Learning & EdTech': ['e-learning', 'MOOCs', 'virtual classroom', 'online courses', 'EdTech'],

    # 🏛️ Government & Policy
    'Regulations & Compliance': [
        'regulation', 'compliance', 'laws', 'policy', 'audit', 'standards',
        'cybersecurity compliance', 'government regulations', 'data privacy laws'
    ],
    'Politics & International Relations': ['politics', 'government', 'elections', 'diplomacy', 'foreign policy'],
    'Government Policies & Global Agreements': ['United Nations', 'developing countries', 'global cooperation'],

    # 🌍 Society & Ethics
    'Diversity & Inclusion': ['diversity', 'inclusion', 'gender equality', 'racism', 'social justice'],
    'Philosophy & Ethics': ['philosophy', 'morality', 'ethics', 'values', 'debates'],
    'Legal & Criminal Justice': ['law', 'justice', 'court', 'crime', 'legal system'],

    # 💡 Personal Development & Lifestyle
    'Self-Improvement & Productivity': ['self-improvement', 'motivation', 'goal setting', 'habits', 'efficiency', 'workflow'],
    'Fitness & Wellness': ['fitness', 'exercise', 'yoga', 'meditation', 'nutrition'],
    'Travel & Adventure': ['travel', 'tourism', 'adventure', 'hotels'],

    # 🎭 Entertainment & Media
    'Movies & TV Shows': ['movies', 'TV shows', 'cinema', 'streaming', 'actors', 'directors'],
    'Music & Performing Arts': ['music', 'concerts', 'albums', 'orchestra', 'theater'],
    'Gaming & Esports': ['gaming', 'esports', 'video games', 'VR', 'tournaments'],

    # 🚘 Transportation & Automotive
    'Automobile & Electric Vehicles': ['cars', 'EVs', 'batteries', 'hybrid cars', 'self-driving'],
    'Aviation & Aerospace': ['aviation', 'airlines', 'drones', 'rockets'],
    'Maritime & Shipping': ['shipping', 'cargo', 'logistics'],

    # 🏡 Home & Lifestyle
    'Interior Design & Home Improvement': ['interior design', 'renovation', 'home decor'],
    'Food & Culinary Arts': ['cooking', 'recipes', 'restaurants'],
    'Fashion & Beauty': ['fashion', 'clothing', 'skincare', 'makeup'],

    # 🧠 Psychology & Behavior
    'Psychology & Mental Health': ['psychology', 'behavior', 'cognition', 'therapy'],
    'Neuroscience & Cognitive Science': ['neuroscience', 'brain', 'learning'],

    # 🛠️ Miscellaneous
    'Mythology & Folklore': ['myths', 'legends', 'folklore'],
    'History & Archaeology': ['history', 'archaeology', 'ancient civilizations'],
    'Mystery & Conspiracy Theories': ['conspiracy', 'mystery', 'UFOs', 'secret societies']
}

        topic_scores = defaultdict(int)
        for token in doc:
            for topic, keywords in topic_keywords.items():
                if token.text.lower() in keywords:
                    topic_scores[topic] += 1

        return max(topic_scores.items(), key=lambda x: x[1])[0] if topic_scores else 'General Discussion'

    def remove_redundancies(self, chunks: List[str]) -> List[str]:
        """Remove redundant content using TF-IDF and cosine similarity"""
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(chunks)
        similarities = cosine_similarity(tfidf_matrix)

        unique_chunks = []
        for i, chunk in enumerate(chunks):
            is_redundant = False
            for j in range(i):
                if similarities[i][j] > 0.7:
                    is_redundant = True
                    break
            if not is_redundant:
                unique_chunks.append(chunk)

        return unique_chunks

    def format_markdown(self, grouped_chunks: Dict[str, List[str]], meeting_duration: str = None, name_standardization_report: List[str] = None) -> str:
        """Format the grouped chunks into a markdown document with meeting duration"""
        md_content = ["# Meeting Summary\n"]

        # Add meeting duration if available
        if meeting_duration:
            md_content.append(f"**Meeting Duration:** {meeting_duration}\n")

        # Add overview section
        md_content.append("## Overview")
        topics = list(grouped_chunks.keys())
        md_content.append("This meeting covered the following topics:")
        for topic in topics:
            if topic:
                md_content.append(f"- {topic}")
        md_content.append("")

        # Add detailed sections
        for topic, chunks in grouped_chunks.items():
            if not topic:
                continue

            md_content.append(f"## {topic}")
            topic_text = ' '.join(chunks)
            final_summary = self.post_processor.postprocess_summary(topic_text)
            sentences = sent_tokenize(final_summary)

            for sentence in sentences:
                if sentence.strip():
                    md_content.append(f"- {sentence.strip()}")
            md_content.append("")

        md_content.append("## Next Steps")
        md_content.append("- [ ] Review shared materials")
        md_content.append("- [ ] Confirm availability for the next meeting")
        md_content.append("- [ ] Share summary with stakeholders\n")

        return "\n".join(md_content)


class PDFGenerator:
    def __init__(self):
        # Initialize styles for reportlab
        self.styles = getSampleStyleSheet()
        # Create custom styles without using .add() to avoid conflicts
        self.heading1_style = ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=12,
        )
        self.heading2_style = ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=18,
            spaceAfter=10,
        )
        self.bullet_style = ParagraphStyle(
            name='CustomBullet',
            parent=self.styles['Normal'],
            leftIndent=20,
            firstLineIndent=0,
            spaceBefore=2,
            bulletIndent=10,
        )

    def sanitize_text(self, text):
        """Replace problematic Unicode characters with ASCII alternatives"""
        replacements = {
            '•': '*',      # bullet
            '–': '-',      # en dash
            '—': '--',     # em dash
            '"': '"',      # left double quote
            '"': '"',      # right double quote
            ''': "'",      # left single quote
            ''': "'",      # right single quote
            '…': '...',    # ellipsis
            '☑': '[x]',    # checked checkbox
            '□': '[ ]',    # empty checkbox
            '→': '->',     # right arrow
            '⇒': '=>',     # double right arrow
            '✓': 'v',      # check mark
            '✗': 'x',      # cross mark
            '≈': '~',      # approximately equal
            '≠': '!=',     # not equal
            '≤': '<=',     # less than or equal
            '≥': '>=',     # greater than or equal
        }

        for char, replacement in replacements.items():
            text = text.replace(char, replacement)

        # Handle any remaining non-ASCII characters
        return ''.join(c if ord(c) < 128 else '_' for c in text)

    def markdown_to_pdf_reportlab(self, markdown_text):
        """Convert markdown to PDF using ReportLab (supports all Unicode)"""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
        story = []

        lines = markdown_text.split('\n')
        list_items = []
        in_list = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    # End the list
                    story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20, bulletFontSize=12))
                    list_items = []
                    in_list = False
                story.append(Spacer(1, 6))
                continue

            # Process headings
            if line.startswith('# '):
                if in_list:
                    # End the list first
                    story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20, bulletFontSize=12))
                    list_items = []
                    in_list = False
                story.append(Paragraph(line[2:], self.heading1_style))

            elif line.startswith('## '):
                if in_list:
                    # End the list first
                    story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20, bulletFontSize=12))
                    list_items = []
                    in_list = False
                story.append(Paragraph(line[3:], self.heading2_style))

            # Process list items
            elif line.startswith('- ') or line.startswith('* '):
                content = line[2:]

                # Handle checkboxes
                if content.startswith('[ ] '):
                    content = '☐ ' + content[4:]  # Empty checkbox
                elif content.startswith('[x] ') or content.startswith('[X] '):
                    content = '☑ ' + content[4:]  # Checked checkbox

                list_items.append(ListItem(Paragraph(content, self.styles['Normal'])))
                in_list = True

            # Regular paragraph text
            else:
                if in_list:
                    # End the list first
                    story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20, bulletFontSize=12))
                    list_items = []
                    in_list = False

                # Handle bold and italic text
                if '**' in line or '*' in line:
                    # Simple formatting: convert **text** to <b>text</b>
                    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                    # Convert *text* to <i>text</i> but avoid replacing inside already processed tags
                    line = re.sub(r'(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)', r'<i>\1</i>', line)

                story.append(Paragraph(line, self.styles['Normal']))

        # Don't forget to add the last list if we're still in one
        if in_list and list_items:
            story.append(ListFlowable(list_items, bulletType='bullet', leftIndent=20, bulletFontSize=12))

        # Build the PDF
        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data

    def markdown_to_pdf(self, markdown_text):
        """Convert markdown to PDF using ReportLab with fallbacks"""
        try:
            # Try using reportlab first (best quality, supports Unicode)
            return self.markdown_to_pdf_reportlab(markdown_text)
        except Exception as e:
            st.warning(f"Error with ReportLab PDF generation: {str(e)}. Trying with sanitized text...")
            try:
                # Try again with sanitized text
                sanitized_text = self.sanitize_text(markdown_text)
                return self.markdown_to_pdf_reportlab(sanitized_text)
            except Exception as e2:
                st.warning(f"Error with sanitized ReportLab generation: {str(e2)}. Using basic FPDF...")
                # Last resort: use basic FPDF with ASCII only
                return self._create_simple_pdf(markdown_text)

    def _create_simple_pdf(self, markdown_text):
        """Fallback method with maximum compatibility"""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Sanitize the text to ensure it's ASCII-only
        lines = self.sanitize_text(markdown_text).split('\n')

        for line in lines:
            # Skip empty lines
            if not line.strip():
                pdf.ln(5)
                continue

            # Process headers and text with simple formatting
            if line.startswith('# '):
                pdf.set_font("Arial", 'B', 24)
                pdf.multi_cell(0, 10, line[2:])
                pdf.ln(5)
            elif line.startswith('## '):
                pdf.set_font("Arial", 'B', 18)
                pdf.multi_cell(0, 10, line[3:])
                pdf.ln(3)
            elif line.startswith('- ') or line.startswith('* '):
                pdf.set_font("Arial", '', 12)
                content = line[2:]
                pdf.multi_cell(0, 5, "- " + content)
            else:
                pdf.set_font("Arial", '', 12)
                pdf.multi_cell(0, 5, line)

        try:
            return pdf.output(dest='S').encode('latin1')
        except Exception as e:
            st.error(f"Critical PDF generation error: {str(e)}")
            # Create an absolutely minimal PDF as last resort
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(40, 10, 'Meeting Summary')
            pdf.ln(15)
            pdf.set_font("Arial", '', 12)
            pdf.cell(0, 10, 'Please download the Markdown version instead.')
            return pdf.output(dest='S').encode('latin1')

    def create_download_link(self, pdf_bytes, filename):
        """Create a download link for the PDF"""
        b64 = base64.b64encode(pdf_bytes).decode()
        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF</a>'
        return href


def create_streamlit_app():


    st.title("🎙️MeetingMind🧠: Meeting Audio to Summary Converter")
    st.write("Upload an audio file to generate a detailed meeting summary")

    # Initialize session state for storing processors
    if 'processors_initialized' not in st.session_state:
        st.session_state.processors_initialized = False
        st.session_state.audio_processor = None
        st.session_state.text_preprocessor = None
        st.session_state.text_summarizer = None
        st.session_state.pdf_generator = None

    # Initialize processors if not already done
    if not st.session_state.processors_initialized:
        with st.spinner("Initializing models... This may take a few minutes..."):
            st.session_state.audio_processor = AudioProcessor()
            st.session_state.text_preprocessor = TextPreprocessor()
            st.session_state.text_summarizer = TextSummarizer()
            st.session_state.pdf_generator = PDFGenerator()
            st.session_state.processors_initialized = True

    # Add explanation about name standardization feature
    with st.expander("ℹ️ About Name Standardization"):
        st.write("""
        This application now includes automatic name standardization to handle misspelled or
        mispronounced names in meeting transcripts. The system uses phonetic algorithms (Double Metaphone)
        to identify when different spellings likely refer to the same person, ensuring consistent
        references throughout your summary.

        **How it works:**
        1. The system detects potential names in the transcript
        2. Names that sound similar are grouped together
        3. The most likely correct spelling is chosen as the standard
        4. All variants are replaced with the standard spelling

        You'll see a section in your final summary that lists all standardized names.
        """)

    # File uploader
    audio_file = st.file_uploader("Choose an audio file", type=['mp3', 'wav', 'm4a'])

    if audio_file is not None:
        # Add options for name standardization
        name_standardization = st.checkbox("Enable name standardization", value=True,
                                          help="Automatically detect and standardize names that sound similar but are spelled differently")

        if st.button("Generate Summary"):
            try:
                # Create placeholder for progress bar and status
                progress_placeholder = st.empty()
                status_text = st.empty()

                # Save uploaded file temporarily
                temp_path = f"temp_audio{Path(audio_file.name).suffix}"
                with open(temp_path, "wb") as f:
                    f.write(audio_file.getbuffer())

                # Process the audio file
                with progress_placeholder.container():
                    # Get audio duration first
                    status_text.text("⏱️ Analyzing audio duration...")
                    audio_duration = st.session_state.audio_processor.get_audio_duration(temp_path)
                    formatted_duration = st.session_state.audio_processor.format_duration(audio_duration)

                    # Step 1: Transcribe audio
                    status_text.text("📝 Transcribing audio... This may take a while...")
                    progress_bar = st.progress(0)
                    transcript = st.session_state.audio_processor.transcribe(temp_path)
                    progress_bar.progress(25)

                    # Save original transcript for comparison if needed
                    original_transcript = transcript

                    # Step 2: Extract potential names (if enabled)
                    if name_standardization:
                        status_text.text("👤 Detecting and standardizing names...")
                        # Pre-extract names to build the standardization dictionary
                        st.session_state.text_preprocessor.name_standardizer.extract_names_from_text(transcript)

                    # Step 3: Preprocess transcript
                    status_text.text("🔍 Preprocessing transcript...")
                    processed_text = st.session_state.text_preprocessor.preprocess(transcript)
                    progress_bar.progress(40)

                    # Get name standardization report if enabled
                    name_report = []
                    if name_standardization:
                        name_report = st.session_state.text_preprocessor.get_name_standardization_report()

                    # Step 4: Split into chunks
                    status_text.text("✂️ Splitting into chunks...")
                    chunks = st.session_state.text_preprocessor.split_into_chunks(processed_text)
                    progress_bar.progress(50)

                    # Step 5: Remove redundancies
                    status_text.text("🔄 Removing redundancies...")
                    unique_chunks = st.session_state.text_summarizer.remove_redundancies(chunks)
                    progress_bar.progress(60)

                    # Step 6: Process chunks and generate summaries
                    status_text.text("📊 Generating summaries...")
                    chunk_progress = st.progress(0)
                    grouped_chunks = st.session_state.text_summarizer.process_chunks(unique_chunks, chunk_progress)
                    progress_bar.progress(80)

                    # Step 7: Format summary
                    status_text.text("📑 Formatting summary...")
                    markdown_content = st.session_state.text_summarizer.format_markdown(
                        grouped_chunks,
                        formatted_duration,
                        name_report if name_standardization else None
                    )
                    progress_bar.progress(90)

                    # Step 8: Refine summary
                    status_text.text("✨ Refining summary.")
                    try:
                        client = openai.OpenAI(
                            api_key=GITHUB_TOKEN,
                            base_url=BASE_URL
                        )

                        response = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "system",
                                    "content": """You are an expert meeting summary editor. Transform raw meeting summaries into well-structured, clear documents. Focus on:
                        1. Enhancing readability and logical flow
                        2. Eliminating redundancies - remove repeated information and consolidate similar points
                        3. Organizing content into clearly defined sections
                        4. Extracting and highlighting:
                           - Meeting agenda points
                           - Action items with owners and deadlines
                           - Required follow-ups
                           - Suggested topics for future meetings
                        5. Preserving meeting duration and all unique, non-redundant information
                        You may reduce content length, but only by removing duplicated information. Do not remove unique details."""
                                },
                                {
                                    "role": "user",
                                    "content": f"Refine this meeting summary to improve clarity and organization. Remove any repeated information while keeping all unique content.\n\nSUMMARY:\n{markdown_content}"
                                }
                            ],
                            model=MODEL,
                            temperature=0.5,
                            max_tokens=10000
                        )
                        refined_summary = response.choices[0].message.content.strip()
                    except Exception as e:
                        st.warning(f"Could not refine summary with GPT-4: {str(e)}. Using original summary.")
                        refined_summary = markdown_content

                    progress_bar.progress(100)
                    status_text.text("✅ Summary generated successfully!")

                    # Generate PDF from the refined summary
                    try:
                        pdf_bytes = st.session_state.pdf_generator.markdown_to_pdf(refined_summary)
                    except Exception as e:
                        st.warning(f"PDF generation encountered an error: {str(e)}. Trying fallback method...")
                        try:
                            # Use the fallback method directly
                            pdf_bytes = st.session_state.pdf_generator._create_simple_pdf(refined_summary)
                        except Exception as e2:
                            st.error(f"Failed to generate PDF with fallback method: {str(e2)}")
                            pdf_bytes = None

                    # Display results in tabs
                    tab1, tab2, tab3 = st.tabs(["📝 Summary", "📊 Details", "🔍 Name Standardization"])

                    with tab1:
                        st.markdown(refined_summary)

                        # Add download buttons
                        col1, col2 = st.columns(2)

                        with col1:
                            st.download_button(
                                label="📥 Download Markdown",
                                data=refined_summary,
                                file_name="meeting_summary.md",
                                mime="text/markdown"
                            )

                        with col2:
                            if pdf_bytes:
                                st.download_button(
                                    label="📥 Download PDF",
                                    data=pdf_bytes,
                                    file_name="meeting_summary.pdf",
                                    mime="application/pdf"
                                )
                            else:
                                st.error("PDF generation failed. Please download the Markdown version instead.")

                    with tab2:
                        st.subheader("Processing Statistics")
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Meeting Duration", formatted_duration)
                        with col2:
                            st.metric("Original Chunks", len(chunks))
                        with col3:
                            st.metric("After Deduplication", len(unique_chunks))
                        with col4:
                            st.metric("Topics Identified", len(grouped_chunks))

                        st.subheader("Identified Topics")
                        for topic in grouped_chunks.keys():
                            st.write(f"- {topic}")

                    with tab3:
                        st.subheader("Name Standardization Results")

                        if name_standardization:
                            name_groups = st.session_state.text_preprocessor.name_standardizer.get_name_groups()
                            standardized_names = st.session_state.text_preprocessor.name_standardizer.get_name_variants()

                            # Show statistics
                            total_names = len(standardized_names)
                            unique_names = len(name_groups)

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Total Name Mentions", total_names)
                            with col2:
                                st.metric("Unique Names (After Standardization)", unique_names)

                            # Show groups with multiple variants
                            st.subheader("Name Standardization Groups")

                            has_groups = False
                            for metaphone, names in name_groups.items():
                                if len(names) > 1:  # Only show groups with variants
                                    has_groups = True
                                    standard_name = st.session_state.text_preprocessor.name_standardizer.metaphone_mapping.get(metaphone, list(names)[0])
                                    variants = [name for name in names if name != standard_name]
                                    if variants:  # Only report if there are actual variants
                                        st.success(f"**{standard_name}** ← {', '.join(variants)}")

                            if not has_groups:
                                st.info("No name variants requiring standardization were detected.")

                            # Option to see example snippets
                            if st.checkbox("Show name occurrence examples from transcript"):
                                st.subheader("Name Occurrences in Transcript")

                                # Only display prominent names (appearing multiple times)
                                prominent_names = [name for metaphone, names in name_groups.items()
                                                for name in names
                                                if len(re.findall(r'\b' + re.escape(name) + r'\b', original_transcript)) > 1]

                                for name in sorted(set(prominent_names)):
                                    # Find occurrences with surrounding context
                                    matches = re.finditer(r'\b' + re.escape(name) + r'\b', original_transcript)
                                    examples = []
                                    for match in matches:
                                        start = max(0, match.start() - 50)
                                        end = min(len(original_transcript), match.end() + 50)
                                        context = original_transcript[start:end]

                                        # Highlight the name
                                        highlighted = context.replace(name, f"**{name}**")
                                        examples.append(highlighted)

                                        if len(examples) >= 2:  # Limit to 2 examples per name
                                            break

                                    if examples:
                                        with st.expander(f"Examples of '{name}'"):
                                            for i, example in enumerate(examples):
                                                st.markdown(f"Example {i+1}: \"...{example}...\"")
                        else:
                            st.info("Name standardization was not enabled for this summary. Enable it when generating a summary to see results here.")

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.error("Please try again with a different audio file or check the file format.")

            finally:
                # Cleanup
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                torch.cuda.empty_cache()
                gc.collect()

    # Add sidebar with information
    with st.sidebar:
        st.subheader("ℹ️ About")
        st.write("""
        This app converts meeting audio to detailed summaries with intelligent name standardization
        to handle misspelled or mispronounced names.
        """)

        st.subheader("📋 Supported Formats")
        st.write("- MP3\n- WAV\n- M4A")

        st.subheader("🔤 Name Standardization")
        st.write("""
        The app uses the Double Metaphone algorithm to identify phonetically similar names that likely refer
        to the same person, even when spelling varies due to speech recognition errors or mispronunciation.

        This helps ensure that the same person is consistently referenced throughout the summary.
        """)

        st.subheader("⚙️ System Status")
        st.write(f"Using device: {device}")
        if torch.cuda.is_available():
            st.write(f"GPU: {torch.cuda.get_device_name(0)}")



# Main entry point
if __name__ == "__main__":
    create_streamlit_app()
