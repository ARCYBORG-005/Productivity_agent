
# StudyAgent | AI Learning and Productivity Dashboard

StudyAgent is a personal learning and productivity app for organizing study sessions, tracking coding practice, and preparing for technical interviews. It combines a Streamlit dashboard, MongoDB persistence, and optional AI services to help learners plan work and review progress.

This project was built as a portfolio project to demonstrate Python application development, database integration, API design, and the use of generative AI and local AI tools.

## Features

- **Learning dashboard:** View a daily study checklist and track completed tasks.
- **AI learning coach:** Ask questions about coding and learning topics through the Gemini integration.
- **Coding coach:** Use a local Ollama model to receive step-by-step explanations of code.
- **Voice learning tools:** Record questions, transcribe audio with Faster-Whisper, and generate spoken responses with Edge TTS.
- **Progress tracking:** Record concept mastery, weak topics, diagnostic sessions, solved problems, and code reviews.
- **Study analytics:** Review practice consistency, completed tasks, and active versus passive study time.
- **Interview preparation:** Track interview readiness and generate company- and role-focused preparation guidance.
- **Persistent storage:** Store learning activity and progress in MongoDB.

Some AI and voice features depend on external services or locally installed models. See [Optional AI and Voice Features](#optional-ai-and-voice-features).

## Technology Stack

- **Language:** Python
- **Dashboard:** Streamlit
- **Backend API:** FastAPI and Uvicorn
- **Database:** MongoDB with PyMongo and Motor
- **Cloud AI:** Google Gemini API (`google-genai`)
- **Local AI:** Ollama
- **Speech:** Faster-Whisper and Edge TTS
- **Data validation:** Pydantic

## Getting Started

### Requirements

- Python 3.11 or later
- MongoDB running locally, or a MongoDB connection URI
- A Gemini API key for Gemini-powered features

### 1. Clone the repository

```bash
git clone https://github.com/<your-github-username>/<your-repository-name>.git
cd <your-repository-name>
```

Replace the placeholders with your GitHub username and repository name.

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install rich
```

`rich` is used by the agent module but is not currently listed in `requirements.txt`, so install it separately. Some audio and machine-learning packages can take extra time and disk space to install.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=productivity_agent_db
```

Keep API keys private. Do not commit `.env` or paste a real key into source code, screenshots, or public issues. MongoDB defaults to the local URI shown above if `MONGO_URI` is omitted.

### 5. Start MongoDB

Start your local MongoDB server, or set `MONGO_URI` to your hosted MongoDB connection string. The application stores checklists, chat history, study progress, and analytics data in MongoDB.

### 6. Run the dashboard

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

Alternatively, run the launcher script:

```bash
python run_app.py
```

## Optional AI and Voice Features

### Gemini

Set `GEMINI_API_KEY` in `.env` to enable features that call the Google Gemini API. API access and usage may be subject to Google account, billing, and quota requirements.

### Ollama coding coach

Install and start [Ollama](https://ollama.com/), then download a model supported by the app. The default coding-coach model is `gemma2:9b`:

```bash
ollama pull gemma2:9b
```

The dashboard also offers `gemma2:2b`, `llama3.2`, and `mistral` as model choices. Local model memory and performance requirements depend on the model you select.

### Voice input and output

The voice workflow uses a microphone, Faster-Whisper for speech transcription, Edge TTS for speech generation, and local audio playback libraries. These features may require additional operating-system audio configuration and model downloads. The core dashboard can be used without voice features.

## Backend API

The repository includes a FastAPI application scaffold in `app/main.py` and a `POST /api/v1/checkin` route intended to accept available study time and an energy rating and return an AI-generated schedule. The route currently imports request models from `app.models`, but that module is not present in the repository, so the API needs that import/model setup completed before it can start. The Streamlit dashboard is the primary runnable application path.

Start the API during development with:

```bash
uvicorn app.main:app --reload
```

After the API startup issue is resolved and `GEMINI_API_KEY` is configured, interactive documentation is available at <http://127.0.0.1:8000/docs> and the health endpoint is <http://127.0.0.1:8000/>.

## CLI Utility

The command-line utility can initialize the default user record in MongoDB:

```bash
python cli.py init-db
```

## Project Structure

```text
.
├── app.py                    # Streamlit dashboard
├── run_app.py                # Dashboard launcher
├── agent.py                  # Gemini agent and tool-calling logic
├── tools.py                  # Learning, checklist, and analytics operations
├── database.py               # MongoDB connection and collections
├── memory.py                 # Memory-related functionality
├── models.py                 # Pydantic data models
├── cli.py                    # Database initialization command
├── app/
│   ├── main.py               # FastAPI application
│   ├── config.py             # API settings
│   └── routes/
│       └── agent_routes.py   # API endpoints
└── requirements.txt          # Python dependencies
```

## Data and Security Notes

- Use a private database for personal information and learning history.
- Never commit `.env` files, API keys, database passwords, or other credentials.
- The app uses a default user identifier for local use and does not provide production-grade user authentication.
- Review AI-generated schedules, explanations, and interview guidance before relying on them.

## Future Improvements

- Add user authentication and per-user settings.
- Add automated tests for the dashboard, agent, and database operations.
- Complete and test the FastAPI request model wiring.
- Add deployment configuration and a production-ready secrets-management approach.
- Improve cross-platform configuration for audio and model cache paths.

## License

No license file is currently included. 
