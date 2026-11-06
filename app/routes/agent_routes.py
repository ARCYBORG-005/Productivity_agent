from fastapi import APIRouter, HTTPException
from datetime import datetime
from google import genai

from app.config import settings
from database import db
from app.models import DailyCheckInRequest, TaskLogRequest

router = APIRouter(prefix="/api/v1", tags=["Agent Operations"])

def get_ai_client():
    if not settings.GEMINI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="GEMINI_API_KEY is not configured in .env!"
        )
    return genai.Client(api_key=settings.GEMINI_API_KEY)


@router.post("/checkin")
async def daily_checkin(checkin: DailyCheckInRequest):
    """Generates an adaptive schedule based on daily time and energy."""
    ai_client = get_ai_client()
    
    profile_cursor = db.skill_profile.find({}, {"_id": 0})
    skills_profile = await profile_cursor.to_list(length=10)

    if not skills_profile:
        skills_profile = [
            {"skill": "python_dsa", "multiplier": 1.0, "load": "Heavy"},
            {"skill": "sql", "multiplier": 1.0, "load": "Medium"},
            {"skill": "hindi", "multiplier": 1.0, "load": "Light"},
            {"skill": "english", "multiplier": 1.0, "load": "Light"},
        ]
        await db.skill_profile.insert_many(skills_profile)

    prompt = f"""
    You are an AI personal learning tutor.
    User's state today:
    - Available Time: {checkin.available_minutes} minutes
    - Energy Level (1-5): {checkin.energy_level}/5
    
    Skill Profiles:
    {skills_profile}
    
    Rules:
    1. If energy <= 2, avoid 'Heavy' tasks (like DSA). Prioritize 'Light' or 'Medium' skills.
    2. Total time must strictly stay under {checkin.available_minutes} minutes.
    3. Format output in concise, structured bullet points with minute allocations.
    """

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )

        today_plan = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "available_minutes": checkin.available_minutes,
            "energy_level": checkin.energy_level,
            "plan": response.text,
            "created_at": datetime.now()
        }
        await db.daily_plans.insert_one(today_plan)

        return {
            "status": "success",
            "schedule": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")