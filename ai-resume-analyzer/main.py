import os
import json
import re
import fitz  # PyMuPDF
import shutil
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import google.generativeai as genai

# --- Configuration ---
# The variable MUST be named 'app' for Vercel to detect it.
app = FastAPI(title="AI Resume Analyzer")

# Configure the Gemini API key
try:
    API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=API_KEY)
except Exception as e:
    # This will help debug if the API key is not set
    print(f"Error configuring Gemini API: {e}")

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

# --- Helper Functions (from old utils.py) ---
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def analyze_resume_with_ai(resume_text: str, job_desc: str) -> dict:
    """Analyzes resume with Gemini AI and returns structured JSON."""
    prompt = f"""
    You are a career advisor. Compare this resume with the job description.

    Resume:
    {resume_text}

    Job Description:
    {job_desc}

    Return the output STRICTLY in JSON format like this:
    {{
      "match_percentage": 65,
      "strengths": ["Skill 1", "Skill 2"],
      "weaknesses": ["Weakness 1", "Weakness 2"]
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-pro') # Using gemini-pro which is stable
        response = model.generate_content(prompt)
        content = response.text

        # Extract JSON inside ```json ... ``` if present
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        return json.loads(content)
    except Exception as e:
        print(f"Error during AI analysis: {e}")
        return {"error": "Failed to get a valid response from the AI model.", "details": str(e)}

# --- API Routes ---
@app.get("/api/test")
async def test_route():
    """A simple test route to check if the backend is running."""
    return {"message": "Python backend is alive!"}

@app.post("/api/analyze_resume")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save PDF temporarily
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        resume_text = await run_in_threadpool(extract_text_from_pdf, file_path)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        result = await run_in_threadpool(analyze_resume_with_ai, resume_text, job_desc)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

