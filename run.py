"""Production entrypoint for FastAPI backend on Render / Railway / local."""
import os
import uvicorn

if __name__ == "__main__":
    # Render and cloud hosts dynamically assign $PORT
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    host = os.getenv("API_HOST", "0.0.0.0")
    print(f"Starting server on {host}:{port} ...")
    uvicorn.run("src.api.server:app", host=host, port=port)