from fastapi import FastAPI
from app.routes.agent_routes import router as agent_router

app = FastAPI(
    title="Personal Adaptive Productivity Agent",
    version="1.0.0",
    description="Backend API powering an AI adaptive learning and productivity coach."
)

# Register routes
app.include_router(agent_router)

@app.get("/")
async def root():
    return {"status": "online", "message": "Productivity Agent API is running."}