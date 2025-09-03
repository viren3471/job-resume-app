from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import shutil
import os
import asyncio

# Import your utils
# Make sure the 'app' folder with 'utils.py' is in the 'ai-resume-analyzer' directory
from app.utils import extract_text_from_pdf, analyze_resume_with_ai

# FastAPI app
# The variable MUST be named 'app' for Vercel to detect it.
app = FastAPI(title="AI Resume Analyzer")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory - Vercel uses a temporary directory, so this is for local testing
UPLOAD_DIR = "/tmp/uploads" # Use /tmp for Vercel's writable directory
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
async def root():
    return {"message": "Backend is running!"}


# ✅ Resume analysis route - THIS IS THE CORRECT ROUTE
@app.post("/api/analyze_resume")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save PDF temporarily
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Extract text in background thread
        resume_text = await run_in_threadpool(extract_text_from_pdf, file_path)
        
        # If PDF text extraction fails
        if not resume_text:
             raise HTTPException(status_code=400, detail="Could not extract text from PDF.")


        # Call AI analysis
        result = await run_in_threadpool(analyze_resume_with_ai, resume_text, job_desc)

    except Exception as e:
        # Catch any other errors during processing
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        # Always delete uploaded PDF after processing
        if os.path.exists(file_path):
            os.remove(file_path)

    return result
