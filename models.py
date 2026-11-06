from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class ConceptNode(BaseModel):
    topic: str
    subtopic: str
    mastery_score: float = Field(0.5, ge=0.0, le=1.0)
    status: str = Field("Learning")
    stuck_points: List[str] = Field(default_factory=list)
    last_practiced: Optional[datetime] = None


class DiagnosticAttempt(BaseModel):
    topic: Optional[str]
    problem_name: str
    user_approach_summary: Optional[str]
    stuck_reason: Optional[str]
    confidence_rating: Optional[int] = Field(None, ge=1, le=5)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DailyCheckInRequest(BaseModel):
    available_minutes: int = Field(..., gt=0)
    energy_level: int = Field(..., ge=1, le=5)


class TaskLogRequest(BaseModel):
    task_name: str
    duration_minutes: int = Field(..., gt=0)
    outcome: Optional[str] = None


class ChatTurn(BaseModel):
    role: str
    parts: List[Dict]
    timestamp: Optional[datetime] = None

class TaskItem(BaseModel):
    task_name: str
    completed: bool = False
    activity_type: str = "Active Practice"  # "Active Practice", "Code Review & Refactoring", "Passive Consumption"
    duration_minutes: int = 0
    notes: Optional[str] = None
    code_reviewed: bool = False

class CodeReviewLog(BaseModel):
    user_id: str = "default_user"
    project_name: str
    lines_reviewed: int = 0
    refactored_summary: str
    edge_cases_tested: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class VideoPracticeLog(BaseModel):
    user_id: str = "default_user"
    video_topic: str
    key_concept: str
    exercise_built_independently: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class SolvedProblemLog(BaseModel):
    user_id: str = "default_user"
    domain: str = "DSA"  # e.g., "DSA", "React", "Python", "SQL", "System Design"
    topic: str  # e.g., "Hashing", "Custom Hooks"
    problem_title: str
    time_taken_minutes: int = 30
    stuck_points: List[str] = Field(default_factory=list)
    optimization_level: str = "Optimal"  # "Naive", "Intermediate", "Optimal"
    needs_revision: bool = False
    revision_note: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class CodeSubmissionReview(BaseModel):
    user_id: str = "default_user"
    domain: str
    topic: str
    task_prompt: str
    submitted_code: str
    correctness_score: int = 8
    optimization_score: int = 8
    clean_code_score: int = 8
    ai_feedback: str
    improved_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ApiUsageRecord(BaseModel):
    user_id: str = "default_user"
    date_str: str
    call_count: int = 0
    estimated_tokens: int = 0
    daily_limit: int = 1500  # Default free tier RPM / RPD reference limit
    last_call: datetime = Field(default_factory=datetime.utcnow)

class DailyChecklist(BaseModel):
    user_id: str = "default_user"
    date_str: str  # Format: "YYYY-MM-DD"
    day_type: str  # "Weekday" or "Weekend"
    tasks: Dict[str, TaskItem] = {}  # e.g., {"DSA": TaskItem(task_name="DSA"), ...}
    completed_all: bool = False
    last_updated: datetime = Field(default_factory=datetime.utcnow)