import os
import json
import re
import fitz  # PyMuPDF
import shutil
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
# We are commenting out the genai import for now to ensure the server starts
# import google.generativeai as genai

# --- Configuration ---
app = FastAPI(title="AI Resume Analyzer")

# --- TEMPORARILY DISABLED API Key Configuration FOR DEBUGGING ---
# We are disabling this to check if the server can start without it.
print("DEBUG_MODE: Skipping Gemini API configuration to test server startup.")
IS_GEMINI_CONFIGURED = False 

# Use /tmp for Vercel's writable directory
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions ---
def extract_text_from_pdf(file_path: str) -> str:
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"ERROR in extract_text_from_pdf: {e}")
        return ""

def analyze_resume_with_ai(resume_text: str, job_desc: str) -> dict:
    # This will now always return an error because the API is disabled
    print("DEBUG_MODE: AI analysis called, but API is disabled.")
    return {"error": "Server is running, but AI analysis is temporarily disabled for testing."}

# --- API Routes ---
@app.get("/api/test")
async def test_route():
    """A simple test route to check if the backend is running."""
    return {"message": "Python backend is alive! (API Disabled)"}

@app.post("/api/analyze_resume")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        resume_text = await run_in_threadpool(extract_text_from_pdf, file_path)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
        # The result will be the error message from the helper function
        result = await run_in_threadpool(analyze_resume_with_ai, resume_text, job_desc)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

