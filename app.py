import os
os.environ["HF_HOME"] = "D:/hf_cache"
os.environ["TORCH_HOME"] = "D:/torch_cache"

# Ensure temp audio directory exists on D: drive
TEMP_AUDIO_DIR = "D:/ProductivityAgent/temp_audio"
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

"""
StudyAgent — Executive AI Dashboard & Unified AI Coding Coach
Run: python -m streamlit run app.py
"""

import streamlit as st
import datetime
import importlib
import json
import time
import asyncio
import requests
import re
from dotenv import load_dotenv

load_dotenv()

import database
import tools

importlib.reload(database)
importlib.reload(tools)

try:
    from streamlit_option_menu import option_menu
    HAS_OPTION_MENU = True
except ImportError:
    HAS_OPTION_MENU = False
    option_menu = None
from database import concepts_col, checklists_col
from tools import (
    log_concept_progress,
    get_weak_spots,
    log_diagnostic_session,
    save_chat_turn,
    load_chat_history,
    clear_chat_history,
    get_or_create_daily_checklist,
    update_checklist_item,
    add_custom_task_to_checklist,
    get_weekly_progress_analytics,
    calculate_interview_confidence,
    prepare_company_interview_roadmap,
    log_code_review_session,
    log_video_practice_bridge,
    log_solved_problem,
    get_topic_summary,
    review_code_submission,
    get_daily_api_usage,
    DEFAULT_USER,
    WEEKDAY_SUBJECTS,
    WEEKEND_SUBJECTS,
)
from agent import generate_with_retry, TOOL_MAP

# Voice & Offline AI Libraries
try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    import numpy as np
    import pygame
    import edge_tts
    from faster_whisper import WhisperModel
except Exception as _import_err:
    pass

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
ACTIVITY_TYPES = ["Active Practice", "Code Review & Refactoring", "Passive Consumption"]
DOMAINS        = ["DSA", "React", "Python", "SQL", "JavaScript", "System Design"]
OPT_LEVELS     = ["Naive", "Intermediate", "Optimal"]
SR_INTERVALS   = [1, 3, 7, 14, 30]          # spaced repetition day intervals

NAV_PAGES = [
    "Dashboard", "AI Coding Coach", "AI Coach", "Daily Checklist", "Analytics",
    "Solved Problems", "Code Reviewer", "Practice Audit",
    "Interview Prep", "Concept Graph", "Diagnostic Log",
]
NAV_ICONS = [
    "house-fill", "code-square", "robot", "check2-square", "bar-chart-line-fill",
    "journal-code", "code-slash", "clipboard2-check",
    "briefcase-fill", "diagram-3-fill", "bug-fill",
]

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="StudyAgent — Executive Dashboard & AI Coding Coach",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# OFFLINE AI & VOICE PIPELINE HELPERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_whisper_model():
    try:
        whisper_dir = os.path.join(os.environ.get("HF_HOME", "D:/hf_cache"), "whisper")
        os.makedirs(whisper_dir, exist_ok=True)
        return WhisperModel("base", device="cpu", compute_type="int8", download_root=whisper_dir)
    except Exception as e:
        st.error(f"Error initializing Faster-Whisper: {e}")
        return None

def transcribe_audio(audio_path: str) -> str:
    model = load_whisper_model()
    if model is None:
        return "Faster-Whisper model unavailable."
    segments, info = model.transcribe(audio_path, beam_size=5)
    return " ".join([segment.text for segment in segments]).strip()

def generate_speech_file(text: str, output_file: str):
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception:
        pass

    async def _gen():
        communicate = edge_tts.Communicate(text, "en-US-AvaNeural")
        await communicate.save(output_file)

    try:
        asyncio.run(_gen())
    except Exception:
        try:
            loop = asyncio.new_event_loop()
            loop.run_until_complete(_gen())
            loop.close()
        except Exception as e:
            st.error(f"TTS generation error: {e}")

def play_audio_async(file_path: str):
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        else:
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
    except Exception as e:
        st.warning(f"Audio playback notice: {e}")

def record_audio_5s(output_file: str, sample_rate: int = 16000, duration: int = 5) -> bool:
    try:
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
        sd.wait()
        wav.write(output_file, sample_rate, audio_data)
        return True
    except Exception as e:
        st.error(f"Microphone recording error: {e}")
        return False

def get_coding_coach_lesson(code_snippet: str, user_prompt: str, model_name: str = "gemma2:9b") -> dict:
    url = "http://127.0.0.1:11434/api/chat"
    system_prompt = (
        "You are an expert AI Coding Coach teaching code line-by-line.\n"
        "Analyze the provided Python code snippet and user question.\n"
        "Output ONLY raw JSON matching this structure (no extra text, no markdown):\n"
        "{\n"
        '  "teacher_speech": "Short 1-2 sentence spoken explanation",\n'
        '  "board_steps": [\n'
        '    {"step": 1, "line": 1, "vars": {"x": 0}, "explanation": "Step explanation"}\n'
        '  ]\n'
        "}\n"
    )
    user_content = f"Python Code Snippet:\n{code_snippet}\n\nUser Question/Prompt: {user_prompt}"
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_k": 20,
            "num_predict": 220
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=90)
        response.raise_for_status()
        res_json = response.json()
        raw_content = res_json.get("message", {}).get("content", "").strip()
        
        # Robust JSON extraction via regex
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        if json_match:
            raw_content = json_match.group(0)
            
        return json.loads(raw_content.strip())
    except requests.exceptions.RequestException as req_err:
        return {
            "teacher_speech": f"Could not connect to local Ollama. Ensure Ollama service is running with {model_name}.",
            "board_steps": [
                {
                    "step": 1,
                    "line": 1,
                    "vars": {"status": "Ollama Connection Error"},
                    "explanation": f"Request error details: {req_err}"
                }
            ]
        }
    except json.JSONDecodeError as json_err:
        return {
            "teacher_speech": "Model response completed.",
            "board_steps": [
                {
                    "step": 1,
                    "line": 1,
                    "vars": {"output": raw_content[:120] if 'raw_content' in locals() else "N/A"},
                    "explanation": f"Note: {json_err}"
                }
            ]
        }

# ---------------------------------------------------------------------------
# THEME & CSS  ── Single source of truth for all design tokens
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  /* ── Design Tokens ───────────────────────────────────────── */
  :root {
    --bg-base:        #07101f;
    --bg-surface:     #0f1d30;
    --bg-raised:      #172540;
    --bg-glass:       rgba(15,29,48,0.85);
    --border:         #1c2e47;
    --border-subtle:  rgba(255,255,255,0.05);
    --text-primary:   #eef4ff;
    --text-secondary: #7a9fc2;
    --text-muted:     #3d5a7a;
    --accent:         #3b82f6;
    --accent-glow:    rgba(59,130,246,0.2);
    --success:        #10b981;
    --warning:        #f59e0b;
    --danger:         #ef4444;
    --purple:         #8b5cf6;
    --radius:         12px;
    --radius-lg:      16px;
    --radius-xl:      20px;
  }

  /* ── Global ──────────────────────────────────────────────── */
  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stMain"],
  .main {
    font-family: 'Inter', system-ui, sans-serif !important;
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
  }
  .main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 4rem;
    max-width: 1300px;
  }

  /* ── Sidebar ─────────────────────────────────────────────── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #040d1a 0%, #081525 100%) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-primary) !important; }

  /* ── Inputs ──────────────────────────────────────────────── */
  .stTextInput > div > div > input,
  .stTextArea  > div > div > textarea,
  .stNumberInput > div > div > input,
  .stSelectbox > div > div,
  .stSelectbox > div > div > div {
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
  }
  .stTextInput > div > div > input:focus,
  .stTextArea  > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
  }

  /* ── Buttons ─────────────────────────────────────────────── */
  .stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #2563eb) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-weight: 600 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(29,78,216,0.35) !important;
  }
  .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 5px 18px rgba(29,78,216,0.45) !important;
  }

  /* ── Tabs ────────────────────────────────────────────────── */
  .stTabs [data-baseweb="tab-list"] {
    background-color: var(--bg-surface) !important;
    border-radius: var(--radius) !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
  }
  .stTabs [data-baseweb="tab"] {
    background-color: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    font-size: 13px !important;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1a3260, #1e40af) !important;
    color: #fff !important;
  }

  /* ── Metrics ─────────────────────────────────────────────── */
  [data-testid="stMetric"] {
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
  }
  [data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
  [data-testid="stMetricValue"] { color: var(--text-primary) !important; font-weight: 700 !important; }

  /* ── Progress bar ────────────────────────────────────────── */
  .stProgress > div > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
    border-radius: 4px !important;
  }
  .stProgress > div > div {
    background-color: var(--bg-raised) !important;
    border-radius: 4px !important;
  }

  /* ── Alert overrides ─────────────────────────────────────── */
  .stAlert { border-radius: var(--radius) !important; }
  .stCheckbox > label { color: var(--text-primary) !important; }

  /* ── Scrollbar ───────────────────────────────────────────── */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Card */
  .card {
    background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-lg);
    padding: 22px 26px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
  }

  /* Page header */
  .page-header {
    background: linear-gradient(135deg, rgba(15,29,48,0.95), rgba(23,37,64,0.85));
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 26px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }
  .page-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #2563eb 0%, #7c3aed 50%, #06b6d4 100%);
  }
  .page-header h2 {
    margin: 0 0 4px 0;
    font-size: 22px;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.025em;
  }
  .page-header p {
    margin: 0;
    font-size: 13px;
    color: var(--text-secondary);
  }

  /* Badge */
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    white-space: nowrap;
  }
  .badge-blue   { background: rgba(59,130,246,0.15);  color: #60a5fa; border: 1px solid rgba(59,130,246,0.25); }
  .badge-green  { background: rgba(16,185,129,0.12);  color: #34d399; border: 1px solid rgba(16,185,129,0.25); }
  .badge-amber  { background: rgba(245,158,11,0.12);  color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
  .badge-red    { background: rgba(239,68,68,0.12);   color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
  .badge-purple { background: rgba(139,92,246,0.12);  color: #a78bfa; border: 1px solid rgba(139,92,246,0.25); }

  /* Stat label */
  .stat-label {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 4px;
  }

  /* Data row */
  .data-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    background-color: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 8px;
    transition: border-color 0.2s ease;
  }
  .data-row:hover { border-color: rgba(59,130,246,0.3); }
  .data-row-title { font-weight: 600; color: var(--text-primary); font-size: 14px; }
  .data-row-meta  { font-size: 12px; color: var(--text-secondary); }

  /* Sidebar brand */
  .sidebar-brand {
    padding: 18px 0 16px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 14px;
  }
  .sidebar-logo {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #1d4ed8, #7c3aed);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    margin-bottom: 10px;
    box-shadow: 0 4px 14px rgba(29,78,216,0.4);
  }
  .sidebar-name {
    font-size: 16px; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.03em;
  }
  .sidebar-sub {
    font-size: 11px; color: var(--text-muted);
    margin-top: 2px;
  }

  /* Sidebar API widget */
  .api-widget {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
  }
  .api-widget-row {
    display: flex; justify-content: space-between;
    font-size: 11px; color: var(--text-secondary);
    font-weight: 500; margin-bottom: 8px;
  }
  .api-track {
    background: var(--bg-raised);
    border-radius: 4px; height: 6px;
    overflow: hidden;
  }
  .api-fill { height: 100%; border-radius: 4px; transition: width 0.4s ease; }
  .api-remaining {
    font-size: 10px; color: var(--text-muted);
    margin-top: 6px;
  }

  /* Metric cards */
  .metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
  }
  .metric-card {
    background: linear-gradient(135deg, rgba(15,29,48,0.98), rgba(23,37,64,0.92));
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 22px 20px;
    position: relative;
    overflow: hidden;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    cursor: default;
  }
  .metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 14px 44px rgba(0,0,0,0.5);
  }
  .metric-top-bar {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: var(--radius-xl) var(--radius-xl) 0 0;
  }
  .metric-glow {
    position: absolute;
    width: 120px; height: 120px;
    border-radius: 50%;
    top: -30px; right: -30px;
    opacity: 0.10;
    filter: blur(28px);
    pointer-events: none;
  }
  .metric-icon { font-size: 24px; margin-bottom: 14px; display: block; }
  .metric-value {
    font-size: 36px; font-weight: 800;
    letter-spacing: -0.04em; line-height: 1;
    margin-bottom: 6px;
  }
  .metric-label {
    font-size: 11px; font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .metric-sub { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

  /* Checklist card */
  .cl-card {
    background: linear-gradient(135deg, rgba(15,29,48,0.98), rgba(23,37,64,0.92));
    border: 1px solid var(--border-subtle);
    border-radius: var(--radius-xl);
    padding: 22px 24px;
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
  }
  .cl-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #10b981, #3b82f6);
    border-radius: var(--radius-xl) var(--radius-xl) 0 0;
  }
  .cl-row {
    display: flex; justify-content: space-between;
    align-items: baseline; margin-bottom: 8px;
  }
  .cl-fraction {
    font-size: 30px; font-weight: 800;
    color: var(--text-primary); letter-spacing: -0.04em;
  }
  .cl-fraction span { font-size: 16px; color: var(--text-secondary); font-weight: 500; }
  .cl-pct { font-size: 18px; font-weight: 700; color: #34d399; }
  .cl-label { font-size: 12px; color: var(--text-secondary); margin-bottom: 14px; }
  .cl-track {
    background: var(--bg-raised);
    border-radius: 6px; height: 8px; overflow: hidden;
  }
  .cl-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #10b981, #3b82f6);
    transition: width 0.7s cubic-bezier(0.4, 0, 0.2, 1);
  }

  /* Notifications */
  .notif-section-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 14px;
  }
  .notif-section-title {
    font-size: 13px; font-weight: 700;
    color: var(--text-primary); flex: 1;
  }
  .notif-badge {
    background: rgba(245,158,11,0.15);
    color: #fbbf24;
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 11px; font-weight: 700;
  }

  .notif-card {
    display: flex; align-items: center; gap: 13px;
    padding: 13px 15px;
    background: linear-gradient(135deg, rgba(15,29,48,0.95), rgba(23,37,64,0.75));
    border: 1px solid var(--border-subtle);
    border-radius: 13px;
    margin-bottom: 10px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
  }
  .notif-card:hover { transform: translateX(2px); }

  .notif-card::before {
    content: '';
    position: absolute; left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 13px 0 0 13px;
  }
  .nc-overdue   { border-color: rgba(245,158,11,0.18); }
  .nc-overdue::before   { background: linear-gradient(180deg, #fbbf24, #f87171); }
  .nc-due-soon  { border-color: rgba(139,92,246,0.18); }
  .nc-due-soon::before  { background: linear-gradient(180deg, #a78bfa, #6366f1); }

  .notif-dot {
    width: 9px; height: 9px;
    border-radius: 50%; flex-shrink: 0;
  }
  .dot-amber {
    background: #fbbf24;
    animation: pulseAmber 2.2s infinite;
  }
  .dot-indigo {
    background: #818cf8;
    animation: pulseIndigo 2.2s infinite;
  }
  @keyframes pulseAmber {
    0%   { box-shadow: 0 0 0 0 rgba(251,191,36,0.55); }
    60%  { box-shadow: 0 0 0 8px rgba(251,191,36,0);  }
    100% { box-shadow: 0 0 0 0 rgba(251,191,36,0);    }
  }
  @keyframes pulseIndigo {
    0%   { box-shadow: 0 0 0 0 rgba(129,140,248,0.55); }
    60%  { box-shadow: 0 0 0 8px rgba(129,140,248,0);  }
    100% { box-shadow: 0 0 0 0 rgba(129,140,248,0);    }
  }

  .notif-body  { flex: 1; min-width: 0; }
  .notif-name  { font-weight: 600; font-size: 13px; color: var(--text-primary); }
  .notif-meta  { font-size: 11px; color: var(--text-secondary); margin-top: 2px; }

  .clear-card {
    background: rgba(16,185,129,0.06);
    border: 1px solid rgba(16,185,129,0.18);
    border-radius: var(--radius-lg);
    padding: 24px;
    text-align: center;
  }

  /* Spaced repetition table */
  .sr-head, .sr-row {
    display: grid;
    grid-template-columns: 1.6fr 1.8fr 1.2fr 1.2fr 0.8fr 1.3fr;
    gap: 8px;
    padding: 10px 16px;
    align-items: center;
  }
  .sr-head {
    background: var(--bg-raised);
    border: 1px solid var(--border);
    border-bottom: none;
    border-radius: 10px 10px 0 0;
  }
  .sr-head span {
    font-size: 10px; font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.07em;
  }
  .sr-row {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-top: none;
    transition: background 0.15s ease;
  }
  .sr-row:last-child { border-radius: 0 0 10px 10px; }
  .sr-row:hover { background: var(--bg-raised); }
  .sr-topic { font-size: 13px; font-weight: 700; color: var(--text-primary); }
  .sr-sub   { font-size: 13px; color: var(--text-primary); }
  .sr-date  { font-size: 12px; color: var(--text-secondary); }

  /* Catch-up banner */
  .catchup-banner {
    background: linear-gradient(135deg, rgba(124,58,237,0.10), rgba(37,99,235,0.08));
    border: 1px solid rgba(124,58,237,0.22);
    border-radius: var(--radius-lg);
    padding: 18px 22px;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 16px;
  }
  .cu-icon  { font-size: 26px; flex-shrink: 0; }
  .cu-title { font-size: 15px; font-weight: 700; color: var(--text-primary); }
  .cu-sub   { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }

  .day-header {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 12px; padding-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .day-pill {
    background: rgba(99,102,241,0.14);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.28);
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 12px; font-weight: 700;
    font-variant-numeric: tabular-nums;
  }
  .day-name { font-size: 13px; color: var(--text-secondary); }

  section[data-testid="stSidebar"] .nav-link {
    font-family: 'Inter', sans-serif !important;
  }
  .streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
  }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SPACED REPETITION HELPERS
# ---------------------------------------------------------------------------
def get_spaced_rep_schedule() -> list:
    concepts = list(concepts_col.find({"user_id": DEFAULT_USER}, {"_id": 0}))
    today    = datetime.date.today()
    schedule = []

    for c in concepts:
        lu = c.get("last_updated")
        if not lu:
            continue
        last_date = lu.date() if isinstance(lu, datetime.datetime) else lu

        idx          = c.get("interval_idx", 0)
        interval_d   = SR_INTERVALS[min(idx, len(SR_INTERVALS) - 1)]
        next_recap   = last_date + datetime.timedelta(days=interval_d)
        days_until   = (next_recap - today).days

        if days_until <= 0:
            status       = "overdue"
            status_label = "Due Today" if days_until == 0 else f"{abs(days_until)}d Overdue"
            dot_cls      = "dot-amber"
            card_cls     = "nc-overdue"
            badge_cls    = "badge-amber"
        elif days_until == 1:
            status       = "due-soon"
            status_label = "Due Tomorrow"
            dot_cls      = "dot-indigo"
            card_cls     = "nc-due-soon"
            badge_cls    = "badge-purple"
        else:
            status       = "upcoming"
            status_label = f"In {days_until}d"
            dot_cls      = ""
            card_cls     = ""
            badge_cls    = "badge-green"

        schedule.append({
            "topic":        c.get("topic", ""),
            "subtopic":     c.get("subtopic", ""),
            "last_reviewed": last_date.strftime("%b %d"),
            "next_recap":   next_recap.strftime("%b %d"),
            "days_until":   days_until,
            "status":       status,
            "status_label": status_label,
            "dot_cls":      dot_cls,
            "card_cls":     card_cls,
            "badge_cls":    badge_cls,
            "interval_idx": idx,
        })

    schedule.sort(key=lambda x: x["days_until"])
    return schedule

def advance_concept_review(topic: str, subtopic: str, interval_idx: int) -> None:
    new_idx = min(interval_idx + 1, len(SR_INTERVALS) - 1)
    concepts_col.update_one(
        {"user_id": DEFAULT_USER, "topic": topic, "subtopic": subtopic},
        {"$set": {
            "last_updated":  datetime.datetime.now(datetime.timezone.utc),
            "interval_idx":  new_idx,
        }},
    )

# ---------------------------------------------------------------------------
# MISSED DAYS HELPERS
# ---------------------------------------------------------------------------
def get_missed_days(lookback_days: int = 30) -> list:
    today     = datetime.date.today()
    date_strs = [
        (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(1, lookback_days + 1)
    ]
    existing  = {
        doc["date_str"]
        for doc in checklists_col.find(
            {"user_id": DEFAULT_USER, "date_str": {"$in": date_strs}},
            {"date_str": 1, "_id": 0},
        )
    }
    return sorted(d for d in date_strs if d not in existing)

def create_retroactive_record(date_str: str, completed_subjects: list, duration: int = 60) -> None:
    dt         = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    is_weekend = dt.weekday() in [5, 6]
    day_type   = "Weekend" if is_weekend else "Weekday"
    subjects   = WEEKEND_SUBJECTS if is_weekend else WEEKDAY_SUBJECTS

    tasks = {
        s: {
            "task_name":       s,
            "completed":       s in completed_subjects,
            "activity_type":   "Active Practice",
            "duration_minutes": duration if s in completed_subjects else 0,
            "notes":           "Retroactively logged",
        }
        for s in subjects
    }
    record = {
        "user_id":       DEFAULT_USER,
        "date_str":      date_str,
        "day_type":      day_type,
        "tasks":         tasks,
        "completed_all": all(t["completed"] for t in tasks.values()),
        "retroactive":   True,
        "last_updated":  datetime.datetime.now(datetime.timezone.utc),
    }
    checklists_col.update_one(
        {"user_id": DEFAULT_USER, "date_str": date_str},
        {"$setOnInsert": record},
        upsert=True,
    )

# ---------------------------------------------------------------------------
# STREAK HELPER
# ---------------------------------------------------------------------------
def calculate_current_streak() -> int:
    today  = datetime.date.today()
    streak = 0
    for i in range(90):
        d_str = (today - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        doc   = checklists_col.find_one(
            {"user_id": DEFAULT_USER, "date_str": d_str}, {"tasks": 1}
        )
        if doc and any(t.get("completed") for t in doc.get("tasks", {}).values()):
            streak += 1
        else:
            break
    return streak

# ---------------------------------------------------------------------------
# REUSABLE UI HELPERS
# ---------------------------------------------------------------------------
def page_header(title: str, subtitle: str = "") -> None:
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="page-header"><h2>{title}</h2>{sub_html}</div>',
        unsafe_allow_html=True,
    )

def data_row(title: str, meta: str = "", badge: str = "", badge_class: str = "badge-blue") -> None:
    badge_html = f'<span class="badge {badge_class}">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <div class="data-row">
            <span class="data-row-title">{title}</span>
            <span class="data-row-meta">{meta} {badge_html}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section_divider(label: str = "") -> None:
    if label:
        st.markdown(
            f'<div class="stat-label" style="margin-top:24px;margin-bottom:12px;">{label}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<hr style='border:none;border-top:1px solid #1c2e47;margin:24px 0;'>",
            unsafe_allow_html=True,
        )

def get_activity_index(task: dict) -> int:
    current = task.get("activity_type", ACTIVITY_TYPES[0])
    return ACTIVITY_TYPES.index(current) if current in ACTIVITY_TYPES else 0

def task_has_changed(task: dict, done: bool, activity: str, duration: int, notes: str) -> bool:
    return (
        done     != task.get("completed", False)
        or activity != task.get("activity_type", ACTIVITY_TYPES[0])
        or duration != task.get("duration_minutes", 0)
        or notes    != task.get("notes", "")
    )

def format_opt_level(raw: str) -> str:
    return {"Naive": "Naive", "Intermediate": "Intermediate", "Optimal": "Optimal"}.get(raw, "Naive")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-logo">🎯</div>
        <div class="sidebar-name">StudyAgent</div>
        <div class="sidebar-sub">Executive AI Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    if HAS_OPTION_MENU and option_menu is not None:
        active_page = option_menu(
            menu_title=None,
            options=NAV_PAGES,
            icons=NAV_ICONS,
            default_index=0,
            styles={
                "container": {"padding": "0", "background-color": "transparent", "border": "none"},
                "icon": {"color": "#3d5a7a", "font-size": "14px"},
                "nav-link": {
                    "font-size": "13px",
                    "color": "#7a9fc2",
                    "border-radius": "9px",
                    "margin": "1px 0",
                    "padding": "9px 14px",
                    "font-family": "'Inter', sans-serif",
                    "font-weight": "500",
                },
                "nav-link-selected": {
                    "background": "linear-gradient(135deg, rgba(29,78,216,0.28), rgba(124,58,237,0.16))",
                    "color": "#dbeafe",
                    "font-weight": "700",
                    "border": "1px solid rgba(59,130,246,0.22)",
                },
                "icon-selected": {"color": "#60a5fa"},
            },
        )
    else:
        active_page = st.radio("Navigation", NAV_PAGES, index=0)

    # API Usage widget
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    api_stats = get_daily_api_usage()
    pct       = min(100.0, api_stats["pct_used"])
    remaining = api_stats["daily_limit"] - api_stats["call_count"]
    bar_color = (
        "linear-gradient(90deg,#10b981,#06b6d4)" if pct < 60
        else "linear-gradient(90deg,#f59e0b,#ef4444)" if pct < 85
        else "linear-gradient(90deg,#ef4444,#dc2626)"
    )
    api_val_color = "#34d399" if pct < 60 else "#fbbf24" if pct < 85 else "#f87171"

    st.markdown(f"""
    <div class="api-widget">
        <div class="api-widget-row">
            <span>API Usage Today</span>
            <span style="color:{api_val_color}; font-weight:700;">
                {api_stats['call_count']} / {api_stats['daily_limit']}
            </span>
        </div>
        <div class="api-track">
            <div class="api-fill" style="width:{pct}%; background:{bar_color};"></div>
        </div>
        <div class="api-remaining">{remaining} requests remaining ({pct}% used)</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick metrics
    weekly = get_weekly_progress_analytics()
    conf   = calculate_interview_confidence()
    c1, c2 = st.columns(2)
    c1.metric("7-Day", f"{weekly['consistency_pct']}%")
    c2.metric("Readiness", f"{conf['confidence_score']}%")

    st.markdown(
        '<div style="margin-top:16px;" class="stat-label">Ollama Gemma 2 · Whisper · Edge-TTS</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# PAGE  ──  DASHBOARD
# ===========================================================================
if active_page == "Dashboard":
    hour = datetime.datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"
    page_header(
        "Executive Dashboard",
        f"{greeting} — here's your complete learning intelligence overview.",
    )

    checklist_today = get_or_create_daily_checklist()
    tasks_today     = checklist_today.get("tasks", {})
    total_tasks     = len(tasks_today)
    done_tasks      = sum(1 for t in tasks_today.values() if t.get("completed"))

    total_concepts  = concepts_col.count_documents({"user_id": DEFAULT_USER})
    streak          = calculate_current_streak()
    api_left        = api_stats["daily_limit"] - api_stats["call_count"]
    readiness       = conf["confidence_score"]

    st.markdown(f"""
    <div class="metrics-grid">
      <div class="metric-card">
        <div class="metric-top-bar" style="background:linear-gradient(90deg,#2563eb,#6366f1);"></div>
        <div class="metric-glow" style="background:#3b82f6;"></div>
        <span class="metric-icon">🧠</span>
        <div class="metric-value" style="color:#60a5fa;">{total_concepts}</div>
        <div class="metric-label">Concepts Tracked</div>
        <div class="metric-sub">In concept knowledge graph</div>
      </div>
      <div class="metric-card">
        <div class="metric-top-bar" style="background:linear-gradient(90deg,#f59e0b,#ef4444);"></div>
        <div class="metric-glow" style="background:#f59e0b;"></div>
        <span class="metric-icon">🔥</span>
        <div class="metric-value" style="color:#fbbf24;">{streak}</div>
        <div class="metric-label">Day Streak</div>
        <div class="metric-sub">Consecutive active days</div>
      </div>
      <div class="metric-card">
        <div class="metric-top-bar" style="background:linear-gradient(90deg,#10b981,#06b6d4);"></div>
        <div class="metric-glow" style="background:#10b981;"></div>
        <span class="metric-icon">⚡</span>
        <div class="metric-value" style="color:#34d399;">{api_left}</div>
        <div class="metric-label">API Calls Left</div>
        <div class="metric-sub">Today's Gemini quota</div>
      </div>
      <div class="metric-card">
        <div class="metric-top-bar" style="background:linear-gradient(90deg,#8b5cf6,#ec4899);"></div>
        <div class="metric-glow" style="background:#8b5cf6;"></div>
        <span class="metric-icon">🏆</span>
        <div class="metric-value" style="color:#a78bfa;">{readiness}%</div>
        <div class="metric-label">Interview Readiness</div>
        <div class="metric-sub">{conf['status_label']}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1.65], gap="large")

    with col_l:
        pct_done  = int(done_tasks / total_tasks * 100) if total_tasks else 0
        day_label = checklist_today.get("day_type", "Weekday")
        st.markdown(f"""
        <div class="cl-card">
            <div class="stat-label" style="margin-bottom:12px;">Today's Study Checklist</div>
            <div class="cl-row">
                <span class="cl-fraction">{done_tasks}<span>/{total_tasks}</span></span>
                <span class="cl-pct">{pct_done}%</span>
            </div>
            <div class="cl-label">Tasks completed &nbsp;·&nbsp; {day_label} track</div>
            <div class="cl-track">
                <div class="cl-fill" style="width:{pct_done}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        schedule = get_spaced_rep_schedule()
        alerts   = [
            s for s in schedule
            if s["status"] in ("overdue", "due-soon")
            and not st.session_state.get(f"ack_{s['topic']}_{s['subtopic']}")
        ]

        if alerts:
            st.markdown(f"""
            <div class="notif-section-header">
                <div class="notif-section-title">Recap Notifications</div>
                <div class="notif-badge">{len(alerts)} pending</div>
            </div>
            """, unsafe_allow_html=True)

            for item in alerts:
                topic_lbl = f"{item['topic'].upper()} / {item['subtopic'].title()}"
                meta_lbl  = f"{item['status_label']}  ·  Last reviewed {item['last_reviewed']}"
                btn_key   = f"btn_review_{item['topic']}_{item['subtopic']}"

                nc_col, btn_col = st.columns([5, 1])
                with nc_col:
                    st.markdown(f"""
                    <div class="notif-card {item['card_cls']}">
                        <div class="notif-dot {item['dot_cls']}"></div>
                        <div class="notif-body">
                            <div class="notif-name">{topic_lbl}</div>
                            <div class="notif-meta">{meta_lbl}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with btn_col:
                    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
                    if st.button("✓", key=btn_key, help="Mark as reviewed", use_container_width=True):
                        advance_concept_review(item["topic"], item["subtopic"], item["interval_idx"])
                        st.session_state[f"ack_{item['topic']}_{item['subtopic']}"] = True
                        st.rerun()
        else:
            st.markdown("""
            <div class="clear-card">
                <div style="font-size:30px; margin-bottom:8px;">✅</div>
                <div style="font-weight:700; color:#34d399; font-size:14px;">All caught up!</div>
                <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">
                    No concepts due for review right now
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_r:
        section_divider("Spaced Repetition Schedule")

        if not schedule:
            st.info("No concepts logged yet. Use the **Concept Graph** page to track your first topic.")
        else:
            st.markdown("""
            <div class="sr-head">
              <span>Topic</span>
              <span>Subtopic</span>
              <span>Last Review</span>
              <span>Next Recap</span>
              <span>Days</span>
              <span>Status</span>
            </div>
            """, unsafe_allow_html=True)

            rows_html = ""
            for item in schedule:
                d = item["days_until"]
                d_color = (
                    "#f87171" if d < 0
                    else "#fbbf24" if d == 0
                    else "#fbbf24" if d == 1
                    else "#94a3b8"
                )
                d_str   = f"{'+' if d > 0 else ''}{d}d"
                badge   = f'<span class="badge {item["badge_cls"]}">{item["status_label"]}</span>'
                rows_html += f"""
                <div class="sr-row">
                  <span class="sr-topic">{item["topic"].upper()}</span>
                  <span class="sr-sub">{item["subtopic"].title()}</span>
                  <span class="sr-date">{item["last_reviewed"]}</span>
                  <span class="sr-date">{item["next_recap"]}</span>
                  <span style="font-size:13px;font-weight:700;color:{d_color};">{d_str}</span>
                  <span>{badge}</span>
                </div>
                """
            st.markdown(rows_html, unsafe_allow_html=True)

    missed = get_missed_days(lookback_days=30)
    if missed:
        st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="catchup-banner">
            <div class="cu-icon">📅</div>
            <div>
                <div class="cu-title">Missed Days — Catch Up</div>
                <div class="cu-sub">
                    {len(missed)} day{'s' if len(missed) > 1 else ''} detected without a study log.
                    Update each day to keep your streak and analytics accurate.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(
            f"📝  Log {len(missed)} Missed Day{'s' if len(missed) > 1 else ''}",
            expanded=(len(missed) <= 5),
        ):
            for date_str in missed:
                dt       = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                day_name = dt.strftime("%A, %B %d %Y")
                is_wknd  = dt.weekday() in [5, 6]
                subjects = WEEKEND_SUBJECTS if is_wknd else WEEKDAY_SUBJECTS

                st.markdown(f"""
                <div class="day-header">
                    <span class="day-pill">{date_str}</span>
                    <span class="day-name">{day_name}</span>
                </div>
                """, unsafe_allow_html=True)

                with st.form(key=f"cu_{date_str}"):
                    comp_subs = st.multiselect(
                        "Subjects you studied that day",
                        options=subjects,
                        key=f"ms_{date_str}",
                        placeholder="Select subjects…",
                    )
                    dur = st.slider(
                        "Avg. time per subject (mins)",
                        min_value=15, max_value=240, value=60, step=15,
                        key=f"sl_{date_str}",
                    )
                    left_btn, _ = st.columns([1, 4])
                    if left_btn.form_submit_button("Save Day", use_container_width=True):
                        create_retroactive_record(date_str, comp_subs, dur)
                        n = len(comp_subs)
                        st.success(f"Logged {n} subject{'s' if n != 1 else ''} for {date_str} ✓")
                        st.rerun()

                st.markdown(
                    "<hr style='border:none;border-top:1px solid #1c2e47;margin:14px 0;'>",
                    unsafe_allow_html=True,
                )


# ===========================================================================
# PAGE  ──  AI CODING COACH (NEW UNIFIED OFFLINE PAGE)
# ===========================================================================
elif active_page == "AI Coding Coach":
    page_header(
        "AI Coding Coach — Live Interactive Blackboard",
        "Offline AI tutoring powered by Ollama (gemma2:9b), Faster-Whisper STT, and Edge-TTS.",
    )

    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="stat-label">Python Code Snippet</div>', unsafe_allow_html=True)
        default_snippet = (
            "def binary_search(arr, target):\n"
            "    left, right = 0, len(arr) - 1\n"
            "    while left <= right:\n"
            "        mid = (left + right) // 2\n"
            "        if arr[mid] == target:\n"
            "            return mid\n"
            "        elif arr[mid] < target:\n"
            "            left = mid + 1\n"
            "        else:\n"
            "            right = mid - 1\n"
            "    return -1\n\n"
            "result = binary_search([1, 3, 5, 7, 9, 11], 7)"
        )
        code_input = st.text_area(
            "Code Editor",
            value=st.session_state.get("coach_code", default_snippet),
            height=260,
            key="coach_code_input",
            help="Enter Python code to analyze or trace line-by-line."
        )

        st.markdown('<div class="stat-label" style="margin-top:16px;">User Question / Voice Input</div>', unsafe_allow_html=True)
        input_mode = st.radio(
            "Input Mode",
            ["Text Prompt", "🎤 5-Second Voice Recording"],
            horizontal=True,
            key="input_mode_radio"
        )

        user_prompt_text = ""

        if input_mode == "Text Prompt":
            user_prompt_text = st.text_input(
                "Question / Instruction",
                value="Trace this code step-by-step and explain how the variables update.",
                placeholder="e.g. Explain line 4 or trace variables...",
                key="coach_text_prompt"
            )
        else:
            rec_col1, rec_col2 = st.columns([1, 1])
            with rec_col1:
                rec_button = st.button("🔴 Record 5s Voice Prompt", use_container_width=True)
            
            audio_path = os.path.join(TEMP_AUDIO_DIR, "user_recording.wav")
            
            if rec_button:
                with st.spinner("🎤 Recording for 5 seconds... Speak now!"):
                    success = record_audio_5s(audio_path, duration=5)
                if success:
                    st.success("Recording complete! Transcribing with Faster-Whisper...")
                    with st.spinner("Transcribing speech..."):
                        try:
                            transcribed = transcribe_audio(audio_path)
                            st.session_state["transcribed_speech"] = transcribed
                        except Exception as e:
                            st.error(f"Whisper transcription failed: {e}")
            
            transcribed_val = st.session_state.get("transcribed_speech", "")
            user_prompt_text = st.text_input(
                "Transcribed Voice Prompt",
                value=transcribed_val,
                placeholder="Your recorded voice prompt will appear here...",
                key="coach_voice_prompt"
            )

        model_choice = st.selectbox("Ollama Model", ["gemma2:9b", "gemma2:2b", "llama3.2", "mistral"], index=0, key="coach_model_choice")

        st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
        run_lesson_btn = st.button("🚀 Run Lesson", type="primary", use_container_width=True)

    with col_right:
        st.markdown('<div class="stat-label">Live Blackboard</div>', unsafe_allow_html=True)
        
        status_placeholder = st.empty()
        speech_box = st.empty()
        active_step_box = st.empty()
        var_state_box = st.empty()

        status_placeholder.markdown("""
        <div class="card" style="border-color:var(--border);">
            <div style="font-weight:700; color:var(--text-secondary);">Ready for lesson</div>
            <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">
                Enter code and a question on the left, then click <b>Run Lesson</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if run_lesson_btn:
            if not user_prompt_text.strip():
                st.warning("Please enter or record a prompt before running the lesson.")
            else:
                status_placeholder.markdown(f"""
                <div class="card" style="border-color:var(--accent);">
                    <div style="font-weight:700; color:#60a5fa;">🧠 Thinking with {model_choice}...</div>
                    <div style="font-size:12px; color:var(--text-secondary); margin-top:4px;">
                        Analyzing code snippet and generating execution steps...
                    </div>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner(f"🧠 {model_choice} is analyzing your code line-by-line..."):
                    lesson_data = get_coding_coach_lesson(code_input, user_prompt_text, model_name=model_choice)
                
                teacher_speech = lesson_data.get("teacher_speech", "Here is the execution walkthrough.")
                board_steps = lesson_data.get("board_steps", [])

                status_placeholder.markdown("""
                <div class="card" style="border-color:var(--purple);">
                    <div style="font-weight:700; color:#a78bfa;">🎙️ Generating Teacher Voice (Edge-TTS)...</div>
                </div>
                """, unsafe_allow_html=True)

                timestamp = int(time.time())
                mp3_path = os.path.join(TEMP_AUDIO_DIR, f"teacher_speech_{timestamp}.mp3")
                generate_speech_file(teacher_speech, mp3_path)

                play_audio_async(mp3_path)

                status_placeholder.markdown("""
                <div class="card" style="border-color:var(--success);">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#34d399;">🔊 Teacher Speaking & Blackboard Active</span>
                        <span class="badge badge-green">LIVE</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                speech_box.markdown(f"""
                <div class="card" style="background:rgba(59,130,246,0.08); border-color:rgba(59,130,246,0.3);">
                    <div style="font-size:11px; font-weight:700; color:var(--accent); text-transform:uppercase; margin-bottom:4px;">Teacher Speech</div>
                    <div style="font-size:14px; font-weight:500; color:var(--text-primary);">{teacher_speech}</div>
                </div>
                """, unsafe_allow_html=True)

                for step_info in board_steps:
                    step_num = step_info.get("step", 1)
                    line_num = step_info.get("line", 1)
                    vars_dict = step_info.get("vars", {})
                    explanation = step_info.get("explanation", "")

                    active_step_box.markdown(f"""
                    <div class="card" style="border-left:4px solid var(--accent); background:var(--bg-raised);">
                        <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                            <span style="font-weight:700; color:#60a5fa;">Step {step_num}</span>
                            <span class="badge badge-purple">Line {line_num}</span>
                        </div>
                        <div style="font-size:13px; color:var(--text-primary);">{explanation}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    var_state_box.markdown('<div class="stat-label">Current Variable States</div>', unsafe_allow_html=True)
                    var_state_box.json(vars_dict)

                    time.sleep(1.8)

                status_placeholder.markdown("""
                <div class="card" style="border-color:var(--success);">
                    <div style="font-weight:700; color:#34d399;">✅ Lesson Complete</div>
                </div>
                """, unsafe_allow_html=True)


# ===========================================================================
# PAGE  ──  AI COACH
# ===========================================================================
elif active_page == "AI Coach":
    page_header(
        "AI Coach",
        "Ask questions about your progress, learning strategy, or interview readiness.",
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        for turn in load_chat_history():
            role  = turn.get("role", "")
            parts = turn.get("parts", [])
            for part in parts:
                if "text" in part:
                    st.session_state.chat_messages.append({
                        "role":    "user" if role == "user" else "assistant",
                        "content": part["text"],
                    })

    _, col_clear = st.columns([8, 1])
    with col_clear:
        if st.button("Clear", key="clear_chat"):
            clear_chat_history()
            st.session_state.chat_messages = []
            st.rerun()

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if not st.session_state.chat_messages:
        st.info("No conversation yet. Type a message below, or click a suggested prompt.")
        col_p1, col_p2, col_p3 = st.columns(3)
        if col_p1.button("Show 7-day progress"):
            st.session_state._queued_prompt = "Show my 7-day consistency and progress"
        if col_p2.button("Check interview readiness"):
            st.session_state._queued_prompt = "How confident am I for technical interviews?"
        if col_p3.button("Generate Amazon SDE roadmap"):
            st.session_state._queued_prompt = "Help me prepare for Amazon Software Development Engineer interview"

    queued     = st.session_state.pop("_queued_prompt", None)
    user_input = st.chat_input("Enter your message…") or queued

    if user_input:
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        history     = load_chat_history()
        user_parts  = [{"text": user_input}]
        history.append({"role": "user", "parts": user_parts})
        save_chat_turn("user", user_parts)

        with st.chat_message("assistant"):
            with st.spinner("Processing…"):
                try:
                    response = generate_with_retry(history, use_tools=True)

                    if getattr(response, "function_calls", None):
                        for call in response.function_calls:
                            tool_name = (
                                call.get("name") if isinstance(call, dict)
                                else getattr(call, "name", None)
                            )
                            tool_args = (
                                call.get("args", {}) if isinstance(call, dict)
                                else getattr(call, "args", {})
                            )
                            st.caption(f"Executed tool: {tool_name}")
                            if tool_name in TOOL_MAP:
                                tool_result = TOOL_MAP[tool_name](**tool_args)
                                st.json(tool_result)
                                call_parts = [{"function_call": {"name": tool_name, "args": tool_args}}]
                                resp_parts = [{"function_response": {"name": tool_name, "response": {"result": tool_result}}}]
                                history.extend([
                                    {"role": "model", "parts": call_parts},
                                    {"role": "user",  "parts": resp_parts},
                                ])
                                save_chat_turn("model", call_parts)
                                save_chat_turn("user",  resp_parts)
                        response = generate_with_retry(history, use_tools=True)

                    reply = getattr(response, "text", "Operation completed.")
                    st.write(reply)
                    save_chat_turn("model", [{"text": reply}])
                    st.session_state.chat_messages.append({"role": "assistant", "content": reply})

                except Exception as error:
                    st.error(f"Error: {error}")


# ===========================================================================
# PAGE  ──  DAILY CHECKLIST
# ===========================================================================
elif active_page == "Daily Checklist":
    checklist = get_or_create_daily_checklist()
    tasks     = checklist.get("tasks", {})
    date_str  = checklist.get("date_str", "")
    day_type  = checklist.get("day_type", "Weekday")

    total     = len(tasks)
    completed = sum(1 for t in tasks.values() if t.get("completed"))
    progress  = completed / total if total else 0.0

    page_header(
        f"Daily Checklist — {date_str}",
        f"{day_type} track  ·  {completed} of {total} tasks completed",
    )

    col_prog, col_metric = st.columns([5, 1])
    with col_prog:
        st.progress(progress)
    with col_metric:
        st.metric("Progress", f"{int(progress * 100)}%")

    if checklist.get("completed_all"):
        st.success("All tasks for today are marked complete. Outstanding work!")

    section_divider("Tasks")

    cols = st.columns(2)
    for index, (task_key, task) in enumerate(tasks.items()):
        with cols[index % 2]:
            task_name = task.get("task_name", task_key)
            with st.container():
                st.markdown(f"**{task_name}**")
                done_val     = st.checkbox("Completed",       value=task.get("completed", False),          key=f"done_{task_key}")
                activity_val = st.selectbox("Activity Type",  ACTIVITY_TYPES,                              index=get_activity_index(task), key=f"act_{task_key}")
                duration_val = st.number_input("Duration (mins)", min_value=0, max_value=600,              value=int(task.get("duration_minutes", 0)), step=15, key=f"dur_{task_key}")
                notes_val    = st.text_input("Notes",          value=task.get("notes", ""),               key=f"notes_{task_key}")
                st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)

            if task_has_changed(task, done_val, activity_val, duration_val, notes_val):
                update_checklist_item(task_key, done_val, notes=notes_val, activity_type=activity_val, duration_minutes=duration_val)
                st.rerun()

    section_divider("Add a Custom Task")
    with st.form("add_task_form"):
        col_name, col_type, col_btn = st.columns([3, 2, 1])
        new_task_name = col_name.text_input("Task Name", placeholder="e.g. React Hooks practice")
        new_task_type = col_type.selectbox("Type", ACTIVITY_TYPES)
        submitted     = col_btn.form_submit_button("Add", use_container_width=True)
        if submitted and new_task_name.strip():
            result = add_custom_task_to_checklist(new_task_name.strip(), activity_type=new_task_type)
            st.success(result)
            st.rerun()


# ===========================================================================
# PAGE  ──  ANALYTICS
# ===========================================================================
elif active_page == "Analytics":
    page_header(
        "Performance Analytics",
        "7-day consistency, active vs passive ratio, and code review health metrics.",
    )

    stats = get_weekly_progress_analytics()

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("7-Day Consistency",    f"{stats['consistency_pct']}%",   f"{stats['days_active']}/7 days active")
    col_b.metric("Active Practice Ratio", f"{stats['active_ratio_pct']}%", "Target >= 75%")
    col_c.metric("Code Review Sessions",  stats["code_reviews_count"])
    col_d.metric("Video-to-Code Bridges", stats["video_bridges_count"])

    if stats["total_passive_mins"] > stats["total_active_mins"]:
        st.error("Passive consumption exceeds active practice. Stop watching tutorials and start building independently.")
    elif stats["active_ratio_pct"] >= 75:
        st.success("Strong active practice ratio. You are spending quality time writing and reviewing code.")

    section_divider("Time Breakdown — Last 7 Days")
    col_e, col_f, col_g = st.columns(3)
    col_e.metric("Active Practice",  f"{stats['total_active_mins']} min")
    col_f.metric("Passive Watching", f"{stats['total_passive_mins']} min")
    col_g.metric("Code Reviews",     f"{stats['total_code_review_mins']} min")

    section_divider("Daily Completion Breakdown")
    for day in stats["daily_breakdown"]:
        col_date, col_bar, col_label = st.columns([2, 5, 2])
        col_date.write(day["date"])
        col_bar.progress(day["rate"] / 100)
        col_label.write(f"{day['completed']}/{day['total']}  ({day['rate']}%)")


# ===========================================================================
# PAGE  ──  SOLVED PROBLEMS
# ===========================================================================
elif active_page == "Solved Problems":
    page_header(
        "Solved Problems Tracker",
        "Log problems you have solved, track time, stuck points, and revision flags.",
    )

    tab_log, tab_view = st.tabs(["Log a Problem", "Topic Summary"])

    with tab_log:
        with st.form("log_problem_form"):
            col_a, col_b, col_c = st.columns(3)
            domain = col_a.selectbox("Domain",         DOMAINS)
            topic  = col_b.text_input("Topic",         placeholder="e.g. Hashing, Custom Hooks")
            title  = col_c.text_input("Problem Title", placeholder="e.g. Two Sum")

            col_d, col_e, col_f = st.columns(3)
            time_taken = col_d.number_input("Time (mins)", min_value=5, max_value=600, value=30, step=5)
            opt_level  = col_e.selectbox("Optimization Level", OPT_LEVELS)
            stuck      = col_f.text_input("Stuck Points", placeholder="e.g. edge cases, off-by-one")

            col_g, col_h = st.columns([1, 3])
            needs_revision = col_g.checkbox("Flag for Revision")
            revision_note  = col_h.text_input("Revision Note", placeholder="e.g. Re-solve using hash map in 3 days")

            if st.form_submit_button("Save Problem", use_container_width=False) and topic and title:
                result = log_solved_problem(
                    topic=topic, problem_title=title, domain=domain,
                    time_taken_minutes=time_taken, optimization_level=format_opt_level(opt_level),
                    stuck_points=stuck, needs_revision=needs_revision, revision_note=revision_note,
                )
                st.success(result["message"])
                st.rerun()

    with tab_view:
        topic_query = st.text_input("Topic to inspect", value="Hashing")
        if topic_query.strip():
            summary = get_topic_summary(topic_query.strip())

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Problems Solved",      summary["total_problems"])
            col_b.metric("Total Time",           f"{summary['total_time_minutes']} min")
            col_c.metric("Flagged for Revision", len(summary["revision_needed_list"]))

            section_divider()
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown('<div class="stat-label">Problems</div>', unsafe_allow_html=True)
                if not summary["problems_list"]:
                    st.info("No problems logged for this topic yet.")
                for problem in summary["problems_list"]:
                    data_row(
                        title=problem["title"],
                        meta=f"{problem['time']} min  |  {problem['level']}  |  {problem['date']}",
                        badge=problem["level"], badge_class="badge-blue",
                    )

            with col_right:
                if summary["revision_needed_list"]:
                    st.markdown('<div class="stat-label">Revision Needed</div>', unsafe_allow_html=True)
                    for item in summary["revision_needed_list"]:
                        data_row(title=item["title"], meta=item["note"], badge="Review", badge_class="badge-amber")
                if summary["all_stuck_points"]:
                    st.markdown('<div class="stat-label" style="margin-top:20px;">Stuck Points</div>', unsafe_allow_html=True)
                    for point in summary["all_stuck_points"]:
                        data_row(title=point, badge_class="badge-red")


# ===========================================================================
# PAGE  ──  CODE REVIEWER
# ===========================================================================
elif active_page == "Code Reviewer":
    page_header(
        "AI Code Reviewer",
        "Submit your solution and receive AI grading on correctness, optimization, and code quality.",
    )

    with st.form("code_review_form"):
        col_a, col_b = st.columns(2)
        domain = col_a.selectbox("Domain", DOMAINS)
        topic  = col_b.text_input("Topic", placeholder="e.g. Custom Hooks, Hashing")
        task_description = st.text_area("Task / Problem Description", placeholder="Describe what the code should do…")
        submitted_code   = st.text_area("Your Solution", height=220, placeholder="# Paste your code here")

        if st.form_submit_button("Submit for Review") and topic and submitted_code:
            with st.spinner("Evaluating solution…"):
                result = review_code_submission(domain, topic, task_description, submitted_code)
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Review complete. Concept graph updated.")
                    review_output = result.get("review_text") or result.get("ai_feedback", "")
                    st.markdown(review_output)
                    if result.get("improved_code"):
                        st.code(result["improved_code"])


# ===========================================================================
# PAGE  ──  PRACTICE AUDIT
# ===========================================================================
elif active_page == "Practice Audit":
    page_header(
        "Practice Audit",
        "Self code review logs and video-to-practice bridge records.",
    )

    tab_review, tab_video = st.tabs(["Self Code Review", "Video Practice Bridge"])

    with tab_review:
        st.info(
            "Before marking any project done: review every line, refactor bad patterns, "
            "and test edge cases. Log the audit here."
        )
        with st.form("self_review_form"):
            project = st.text_input("Project / Module",          placeholder="e.g. User Auth API, React Dashboard")
            lines   = st.number_input("Lines of Code Reviewed",  min_value=1, max_value=5000, value=100)
            summary = st.text_area("What did you refactor?",     placeholder="Extracted reusable hooks, removed duplicate loops…")
            edges   = st.text_input("Edge Cases Tested (comma-separated)", placeholder="empty input, network timeout, index 0")

            if st.form_submit_button("Save Review") and project and summary:
                result = log_code_review_session(project, summary, lines, edges)
                st.success(result["message"])
                st.rerun()

        section_divider("Recent Reviews")
        reviews = list(
            tools.code_reviews_col.find(
                {"user_id": tools.DEFAULT_USER}, {"_id": 0}
            ).sort("timestamp", -1).limit(8)
        )
        if not reviews:
            st.warning("No code reviews logged yet.")
        for review in reviews:
            edges_text = ", ".join(review.get("edge_cases_tested", [])) or "None recorded"
            data_row(
                title=f"{review.get('project_name')}",
                meta=f"{review.get('lines_reviewed')} lines  |  Edge cases: {edges_text}",
            )

    with tab_video:
        st.warning(
            "Rule: for every video you watch, you must build an independent exercise from scratch "
            "without looking at the video. Log it here."
        )
        with st.form("video_bridge_form"):
            video_title    = st.text_input("Video / Course Title",      placeholder="e.g. Node.js Crash Course")
            key_concept    = st.text_input("Key Concept Learned",       placeholder="e.g. JWT Authentication middleware")
            built_exercise = st.text_area("What you built independently", placeholder="Describe the app or script you wrote from scratch…")

            if st.form_submit_button("Save Bridge Log") and video_title and built_exercise:
                result = log_video_practice_bridge(video_title, key_concept, built_exercise)
                st.success(result["message"])
                st.rerun()

        section_divider("Recent Bridges")
        bridges = list(
            tools.video_logs_col.find(
                {"user_id": tools.DEFAULT_USER}, {"_id": 0}
            ).sort("timestamp", -1).limit(8)
        )
        if not bridges:
            st.info("No video-to-practice bridges recorded yet.")
        for bridge in bridges:
            data_row(
                title=bridge.get("video_topic", ""),
                meta=f"Concept: {bridge.get('key_concept', '')}  |  Built: {bridge.get('exercise_built_independently', '')}",
            )


# ===========================================================================
# PAGE  ──  INTERVIEW PREP
# ===========================================================================
elif active_page == "Interview Prep":
    page_header(
        "Interview Readiness",
        "Composite readiness score and targeted company preparation plans.",
    )

    confidence  = calculate_interview_confidence()
    score_color = confidence["status_color"]

    st.markdown(f"""
    <div class="card" style="text-align:center; padding:32px;">
        <div class="stat-label" style="margin-bottom:8px;">Technical Interview Readiness Index</div>
        <div style="font-size:60px; font-weight:800; color:{score_color}; line-height:1;">
            {confidence['confidence_score']}%
        </div>
        <div style="font-size:16px; font-weight:600; color:var(--text-primary); margin-top:8px;">
            {confidence['status_label']}
        </div>
        <div style="font-size:13px; color:var(--text-secondary); margin-top:10px; max-width:640px; margin:10px auto 0;">
            {confidence['advice']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Avg Concept Mastery", f"{confidence['avg_mastery_pct']}%")
    col_b.metric("7-Day Consistency",   f"{confidence['consistency_pct']}%")
    col_c.metric("Open Weak Spots",     confidence["weak_spots_count"])

    section_divider("Generate a Company Preparation Plan")
    with st.form("company_prep_form"):
        col_company, col_role = st.columns(2)
        company = col_company.text_input("Target Company", placeholder="e.g. Google, Amazon, TCS")
        role    = col_role.text_input("Target Role",       placeholder="e.g. Software Development Engineer")
        if st.form_submit_button("Generate Plan") and company and role:
            with st.spinner("Building preparation roadmap…"):
                roadmap = prepare_company_interview_roadmap(company, role)
                section_divider()
                st.markdown(roadmap)


# ===========================================================================
# PAGE  ──  CONCEPT GRAPH
# ===========================================================================
elif active_page == "Concept Graph":
    page_header(
        "Concept Mastery Graph",
        "Track mastery per topic and subtopic, and surface active weak spots.",
    )

    with st.form("log_concept_form"):
        col_a, col_b, col_c = st.columns(3)
        topic    = col_a.text_input("Topic",    placeholder="e.g. DSA")
        subtopic = col_b.text_input("Subtopic", placeholder="e.g. Dynamic Programming")
        status   = col_c.selectbox("Status",   ["Learning", "Mastered", "Stuck"])
        stuck_pt = st.text_input("Stuck Point (optional)", placeholder="e.g. State transition logic")

        if st.form_submit_button("Log Progress") and topic and subtopic:
            result = log_concept_progress(topic, subtopic, status, stuck_pt)
            st.success(f"Updated {topic} / {subtopic} — Mastery score: {result['mastery_score']}")
            st.rerun()

    section_divider("Active Weak Spots")
    weak_spots = get_weak_spots()
    if not weak_spots:
        st.success("No weak spots detected. Keep maintaining your progress.")
    else:
        for spot in weak_spots:
            mastery_pct = int(spot.get("mastery_score", 0.5) * 100)
            label       = f"{spot.get('topic','').upper()} / {spot.get('subtopic','').title()}"
            stuck_list  = ", ".join(spot.get("stuck_points", [])) or "None recorded"

            st.markdown(f"""
            <div class="card" style="border-color:rgba(239,68,68,0.25);">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
                    <span style="font-weight:700;color:var(--text-primary);">{label}</span>
                    <span class="badge badge-red">{mastery_pct}% mastery</span>
                </div>
                <div style="font-size:13px;color:var(--text-secondary);margin-bottom:10px;">
                    Stuck points: {stuck_list}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(mastery_pct / 100)


# ===========================================================================
# PAGE  ──  DIAGNOSTIC LOG
# ===========================================================================
elif active_page == "Diagnostic Log":
    page_header(
        "Diagnostic Problem Log",
        "Record problems where you got stuck and your thought process. "
        "Patterns identified here feed directly into your weak-spot graph.",
    )

    with st.form("diagnostic_form"):
        problem_title   = st.text_input("Problem Title",              placeholder="e.g. LeetCode 15 — 3Sum")
        stuck_reason    = st.text_area("Why you got stuck",           placeholder="e.g. Could not handle duplicate triplets without extra memory.")
        thought_process = st.text_area("Your approach / thought process", placeholder="Walk through every step you attempted…")

        if st.form_submit_button("Save Diagnostic Entry") and problem_title and stuck_reason:
            log_diagnostic_session(problem_title, stuck_reason, thought_process)
            st.success(f"Diagnostic record saved for '{problem_title}'.")
            st.rerun()
