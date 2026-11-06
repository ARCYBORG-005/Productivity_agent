import os
from pathlib import Path
from dotenv import load_dotenv

# Locate the root directory (D:\ProductivityAgent) and load .env from there
BASE_DIR = Path(__file__).resolve().parent.parent
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "productivity_agent_db")

settings = Settings()

# Debug check
if not settings.GEMINI_API_KEY:
    raise ValueError(f"GEMINI_API_KEY is empty! Checked .env path: {env_path}")