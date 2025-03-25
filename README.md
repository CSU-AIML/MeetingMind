# MeetingMind v2.1
> **Developed by :Jay Kanavia**
## 🚀 Overview
MeetingMind v2.1 is an AI-powered meeting summarization tool designed to process meeting transcripts and deliver clear, structured summaries with key points and actionable items. It supports both Google Colab execution and local setup for convenience.

## ✨ Key Features
- Automatic text cleaning and formatting.
- Intelligent extraction of key sentences and discussion highlights.
- Summarization using advanced pre-trained language models.
- Detection and categorization of action items and next steps.
- Export summaries and actions in CSV or text format.

## ✅ Getting Started

### ▶️ Use with Google Colab
1. **Open the notebook in Google Colab.**
2. Navigate to `Runtime > Change runtime type`.
3. Set the hardware accelerator to **GPU** and select **T4 GPU** for best performance.
4. Upload your meeting transcript file.
5. Run all cells sequentially.
6. Download or copy the generated summary and action items.

### 💻 Local Setup Instructions
1. **Clone the repository:**
```bash
git clone <repo_url>
cd MeetingMind-V1-main
```

2. **Install dependencies:**
```bash
python setup.py
```
This will:
- Install Python dependencies listed in `requirements.txt`.
- Download required NLTK datasets (`punkt`, `stopwords`, `words`, `names`).
- Install the spaCy model (`en_core_web_sm`).

3. **Run the application locally:**
```bash
python run.py
```
4. Open the provided local server URL to process meeting transcripts and view summaries.

## 📂 Repository Structure
- `Copy_of_meetingmind_v2_1.ipynb`: The main Colab notebook.
- `setup.py`: One-step installation script for dependencies and language models.
- `requirements.txt`: All Python package dependencies.
- `run.py`: Launch script for the local web application.
- `install_once.py` & `install_fuzzy.py`: Additional helper installation scripts.

## 📊 Output
- Clear summaries with key discussion points.
- Action items listed and categorized by priority.
- Output available for download in CSV or text formats.

## 🤝 Contributing
We welcome contributions! If you have suggestions, enhancements, or bug fixes, please open an issue or submit a pull request.

## 📜 License
This project is licensed under the MIT License.

## 🙏 Acknowledgments
- Hugging Face Transformers
- Sentence Transformers
- NLTK
- spaCy
- Google Colab for providing GPU resources
- Streamlit
---

> **Pro Tip:** Using a **T4 GPU** on Colab significantly speeds up processing times!

> **FOR API KEY CONTACT DEVELOPER , IF YOU ARE PART OF CSU - ONLY THEN SECRETS WILL BE PROVIDED**
