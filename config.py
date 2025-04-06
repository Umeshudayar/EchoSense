import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
LOG_FOLDER = os.path.join(BASE_DIR, 'logs')
MODEL_FOLDER = os.path.join(BASE_DIR, 'models')
DB_FOLDER = os.path.join(BASE_DIR, 'database')

# Create necessary folders if they don't exist
for folder in [UPLOAD_FOLDER, LOG_FOLDER, MODEL_FOLDER, DB_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Database settings
DATABASE_URI = f"sqlite:///{os.path.join(DB_FOLDER, 'transcript_db.sqlite')}"

# ML Model paths
WHISPER_MODEL = "base"  # Options: "tiny", "base", "small", "medium", "large"
VOSK_MODEL_PATH = os.path.join(MODEL_FOLDER, "vosk_model")
EMOTION_MODEL_PATH = os.path.join(MODEL_FOLDER, "emotion_model.pkl")

# API Keys (replace with your actual keys or use environment variables)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "your_openai_key_here")

# Application settings
MAX_AUDIO_LENGTH = 600  # Maximum audio length in seconds (10 minutes)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}