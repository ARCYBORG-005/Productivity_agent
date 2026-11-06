"""
Productivity Agent - AI Learning Strategist & Reasoning Core.
Integrates Gemini LLM function calling with local MongoDB persistence.
"""

import os
import time
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm

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
)

load_dotenv()

console = Console()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    types = None
    GENAI_AVAILABLE = False


TOOL_MAP = {
    "log_concept_progress": log_concept_progress,
    "get_weak_spots": get_weak_spots,
    "log_diagnostic_session": log_diagnostic_session,
    "get_or_create_daily_checklist": get_or_create_daily_checklist,
    "update_checklist_item": update_checklist_item,
    "add_custom_task_to_checklist": add_custom_task_to_checklist,
    "get_weekly_progress_analytics": get_weekly_progress_analytics,
    "calculate_interview_confidence": calculate_interview_confidence,
    "prepare_company_interview_roadmap": prepare_company_interview_roadmap,
    "log_code_review_session": log_code_review_session,
    "log_video_practice_bridge": log_video_practice_bridge,
    "log_solved_problem": log_solved_problem,
    "get_topic_summary": get_topic_summary,
    "review_code_submission": review_code_submission,
    "get_daily_api_usage": get_daily_api_usage,
}


SYSTEM_PROMPT = """
You are an AI Technical Learning Strategist, Elite Code Auditor, and Productivity Coach.

Your Primary Mission:
Eliminate fake productivity (watching videos without practicing, writing unreviewed/untested code) and enforce strict 80/20 active practice & code review habits.

Available Tools:
1. `get_or_create_daily_checklist`: Retrieve or initialize today's checklist.
2. `update_checklist_item`: Mark tasks as completed/pending, specifying activity_type ("Active Practice", "Code Review & Refactoring", "Passive Consumption"), duration_minutes, and code_reviewed status.
3. `add_custom_task_to_checklist`: Dynamically add custom topics or tasks.
4. `log_concept_progress`: Update concept graph mastery scores.
5. `get_weak_spots`: Retrieve user weak topics.
6. `log_diagnostic_session`: Record technical sticking points.
7. `get_weekly_progress_analytics`: Retrieve 1-day and 7-day consistency metrics and Active vs Passive ratio.
8. `calculate_interview_confidence`: Compute overall interview readiness score (0-100%).
9. `prepare_company_interview_roadmap`: Generate targeted company and role preparation guide.
10. `log_code_review_session`: Log mandatory self-code review, refactoring notes, and edge cases tested.
11. `log_video_practice_bridge`: Log conversion of passive video consumption into an active, independent coding exercise built from scratch.

Execution Rules:
- If the user watched a video or tutorial, CHALLENGE them: "Did you build the code independently without looking?" Prompt them to call `log_video_practice_bridge`.
- If the user finished a project or task, CHALLENGE them: "Did you review every line, refactor bad patterns, and test edge cases?" Prompt them to run `log_code_review_session`.
- When updating daily work, track whether it was "Active Practice", "Code Review & Refactoring", or "Passive Consumption".
"""

class SimpleResponse:
    def __init__(self, text: str = "", function_calls: List = None):
        self.text = text
        self.function_calls = function_calls or []


def generate_with_retry(client_or_history, conversation_history=None, retries: int = 3, delay: int = 2, use_tools: bool = True):
    """Execute LLM generation with exponential backoff retry logic."""
    if conversation_history is None:
        conversation_history = client_or_history
        client = None
    else:
        client = client_or_history

    if client is None:
        if not GENAI_AVAILABLE or not GEMINI_API_KEY:
            last_user = ""
            for msg in reversed(conversation_history):
                if msg.get("role") == "user":
                    parts = msg.get("parts", [])
                    if parts and isinstance(parts, list) and "text" in parts[0]:
                        last_user = parts[0]["text"]
                        break
            return SimpleResponse(text=f"[Local Fallback] Message received: {last_user}")

        client = genai.Client(api_key=GEMINI_API_KEY)

    for attempt in range(retries):
        try:
            if use_tools:
                config = None
                if types is not None:
                    config = types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                        tools=list(TOOL_MAP.values()),
                    )
                kwargs = {
                    "model": "gemini-2.0-flash",
                    "contents": conversation_history,
                }
                if config is not None:
                    kwargs["config"] = config
                return client.models.generate_content(**kwargs)

            contents = "\n".join([
                p.get("parts", [{}])[0].get("text", "") if p.get("role") == "user" else ""
                for p in conversation_history
            ])
            return client.models.generate_content(model="gemini-2.0-flash", contents=contents)
        except Exception as e:
            if attempt < retries - 1:
                console.print(f"[dim yellow]API notice: {e}. Retrying in {delay}s...[/dim yellow]")
                time.sleep(delay)
                delay *= 2
            else:
                raise


def run_agent():
    """Run CLI agent interaction loop."""
    console.print(Panel.fit("[bold cyan]Productivity Agent Core Engine[/bold cyan]\n[dim]Connected to MongoDB Concept Graph[/dim]"))
    conversation_history = load_chat_history()

    if conversation_history:
        console.print(f"[bold green]Session history loaded ({len(conversation_history)} records).[/bold green]")
        resume = Confirm.ask("Resume previous session?", default=True)
        if not resume:
            clear_chat_history()
            conversation_history = []
            console.print("[yellow]New session initialized.[/yellow]")

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]User[/bold green]")
            if user_input.strip().lower() in ["exit", "quit"]:
                console.print("[yellow]Session closed.[/yellow]")
                break

            user_parts = [{"text": user_input}]
            user_turn = {"role": "user", "parts": user_parts}
            conversation_history.append(user_turn)
            save_chat_turn("user", user_parts)

            with console.status("[bold blue]Agent processing query...[/bold blue]", spinner="dots"):
                try:
                    response = generate_with_retry(conversation_history, use_tools=True)
                except Exception as err:
                    console.print(f"[bold red]Execution error:[/bold red] {err}")
                    continue

            if getattr(response, "function_calls", None):
                for call in response.function_calls:
                    tool_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                    tool_args = call.get("args", {}) if isinstance(call, dict) else getattr(call, "args", {})
                    console.print(f"[dim yellow]Executed tool: [bold]{tool_name}[/bold] with parameters {tool_args}[/dim yellow]")
                    if tool_name in TOOL_MAP:
                        tool_result = TOOL_MAP[tool_name](**tool_args)
                    else:
                        tool_result = {"error": "tool not found"}

                    call_parts = [{"function_call": {"name": tool_name, "args": tool_args}}]
                    resp_parts = [{"function_response": {"name": tool_name, "response": {"result": tool_result}}}]

                    conversation_history.append({"role": "model", "parts": call_parts})
                    conversation_history.append({"role": "user", "parts": resp_parts})

                    save_chat_turn("model", call_parts)
                    save_chat_turn("user", resp_parts)
                continue

            if getattr(response, "text", None):
                console.print(f"\n[bold cyan]Agent:[/bold cyan]\n{response.text}")
                agent_parts = [{"text": response.text}]
                conversation_history.append({"role": "model", "parts": agent_parts})
                save_chat_turn("model", agent_parts)

        except KeyboardInterrupt:
            console.print("\n\n[bold yellow]Keyboard interrupt detected.[/bold yellow]")
            should_exit = Confirm.ask("Exit application?", default=False)
            if should_exit:
                sys.exit(0)

if __name__ == "__main__":
    run_agent()