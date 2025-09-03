from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import shutil
import os
import asyncio

# This import is the most common point of failure.
# It requires the folder structure to be: ai-resume-analyzer/app/utils.py
from app.utils import extract_text_from_pdf, analyze_resume_with_ai

# The variable MUST be named 'app' for Vercel to detect it.
app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel's only writable directory is /tmp
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ✅ A TEST ROUTE to see if the server is running at all
@app.get("/api/test")
async def test_route():
    return {"message": "Python backend is alive!"}


# ✅ The main analysis route
@app.post("/api/analyze_resume")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        resume_text = await run_in_threadpool(extract_text_from_pdf, file_path)
        
        if not resume_text:
             raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        result = await run_in_threadpool(analyze_resume_with_ai, resume_text, job_desc)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return result

