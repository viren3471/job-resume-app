# All necessary libraries are imported at the top
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool
import shutil
import os
import asyncio
import fitz  # PyMuPDF
import google.generativeai as genai
import json
import re

# ==============================================================================
# HELPER FUNCTIONS (Previously in utils.py)
# ==============================================================================

# Configure the Gemini API key from Vercel's environment variables
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    # This will create a clear error in the Vercel logs if the key is missing
    raise ValueError("GOOGLE_API_KEY environment variable not set in Vercel!")
genai.configure(api_key=API_KEY)


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with fitz.open(file_path) as pdf:
            for page in pdf:
                text += page.get_text()
    except Exception as e:
        print(f"Error reading PDF {file_path}: {e}")
        return ""  # Return empty string on failure
    return text.strip()


def analyze_resume_with_ai(resume_text: str, job_desc: str) -> dict:
    """Analyzes the resume against the job description using the Gemini API."""
    prompt = f"""
You are an expert HR analyst. Your task is to compare the provided resume with the job description and return a detailed analysis.

Resume Text:
---
{resume_text}
---

Job Description:
---
{job_desc}
---

Provide your analysis strictly in the following JSON format. Do not include any text or markdown formatting before or after the JSON object.

{{
  "match_percentage": <A number between 0 and 100 representing the match quality>,
  "strengths": ["A list of key strengths and matching skills from the resume.", "Provide at least two points."],
  "weaknesses": ["A list of skills or requirements from the job description that are missing or not prominent in the resume.", "Provide at least two points."]
}}
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        content = response.text

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            clean_content = json_match.group(1)
        else:
            clean_content = content

        return json.loads(clean_content)

    except json.JSONDecodeError:
        print(f"ERROR: Failed to parse AI response. Raw content: {content}")
        return {"error": "The AI response was not in a valid format. Please try again.", "raw": content}
    except Exception as e:
        print(f"ERROR: An unexpected error occurred with the Gemini API: {e}")
        return {"error": f"An unexpected error occurred: {str(e)}"}


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

app = FastAPI(title="AI Resume Analyzer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "/tmp/uploads"  # Vercel's only writable directory
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
             raise HTTPException(status_code=400, detail="Could not extract text from the PDF. The file might be empty, corrupted, or an image-based PDF.")

        result = await run_in_threadpool(analyze_resume_with_ai, resume_text, job_desc)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        print(f"ERROR in analyze_resume endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

    return result

