import os
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "productivity_agent_db")


def _create_client(uri: str, timeout_ms: int = 2000) -> MongoClient:
    return MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)


try:
    client = _create_client(MONGO_URI)
    # trigger server selection to fail fast if misconfigured
    client.server_info()
except ServerSelectionTimeoutError:
    # fallback to a local default client without raising immediately
    client = MongoClient("mongodb://localhost:27017")


db = client[DB_NAME]

# Common collections used across the project
users_col = db["user_profile"]
daily_plans_col = db["daily_plans"]
concepts_col = db["concept_graph"]
diagnostics_col = db["diagnostics"]
chat_col = db["chat_sessions"]
checklists_col = db["checklists"]
code_reviews_col = db["code_reviews"]
video_logs_col = db["video_practice_logs"]
solved_problems_col = db["solved_problems"]
code_submissions_col = db["code_submissions"]
api_usage_col = db["api_usage"]

def init_db(default_user_id: str = "default_user") -> None:
    """Ensure a minimal default user profile exists in the DB.

    Safe to call multiple times.
    """
    if users_col.count_documents({"user_id": default_user_id}) == 0:
        users_col.insert_one({
            "user_id": default_user_id,
            "created_at": __import__("datetime").datetime.utcnow(),
            "skills": {
                "python_dsa": {"multiplier": 1.0, "status": "In Progress"},
                "sql": {"multiplier": 1.0, "status": "In Progress"},
                "english": {"multiplier": 1.0, "status": "Consistent"},
                "hindi": {"multiplier": 1.0, "status": "Consistent"}
            }
        })


if __name__ == "__main__":
    init_db()