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
app = FastAPI(title="AI Resume Analyzer")

# --- Safer API Key Configuration ---
# This part safely checks for your API key from Vercel's settings.
API_KEY = os.getenv("GOOGLE_API_KEY")
IS_GEMINI_CONFIGURED = False

if not API_KEY:
    # If the key is missing, it will print an error to Vercel logs, but won't crash.
    print("FATAL_ERROR: GOOGLE_API_KEY environment variable not found.")
else:
    try:
        genai.configure(api_key=API_KEY)
        IS_GEMINI_CONFIGURED = True
        # If the key is found and works, it will print a success message.
        print("SUCCESS: Gemini API configured successfully.")
    except Exception as e:
        # If the key is wrong, it will print an error.
        print(f"FATAL_ERROR: Could not configure Gemini API: {e}")

# Use /tmp for Vercel's writable directory. This is important for serverless functions.
UPLOAD_DIR = "/tmp/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- CORS Middleware ---
# This allows your frontend to communicate with your backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper Functions (Logic from your old utils.py is now here) ---

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a given PDF file."""
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
    """Analyzes the resume against the job description using the Gemini API."""
    # First, check if the API key was successfully configured when the server started.
    if not IS_GEMINI_CONFIGURED:
        print("ERROR in analyze_resume_with_ai: Gemini API is not configured.")
        return {"error": "Server-side configuration error: The API Key is missing or invalid."}

    prompt = f"""
    You are an expert HR analyst. Compare the following resume with the provided job description.
    Analyze the resume for skills, experience, and qualifications that match the job requirements.

    Resume Text:
    {resume_text}

    Job Description:
    {job_desc}

    Based on your analysis, return a JSON object with the following structure. Do not include any text or formatting outside of the JSON object itself.
    {{
      "match_percentage": <an integer between 0 and 100 representing the match quality>,
      "strengths": ["A list of key strengths and matching skills from the resume.", "Provide at least two points."],
      "weaknesses": ["A list of areas where the resume is lacking compared to the job description.", "Provide at least two points."]
    }}
    """
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        content = response.text

        # This helps find the JSON even if the AI adds extra text like ```json
        json_match = re.search(r"```json\s*(\{.*?\})\s*```", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        return json.loads(content)
    except Exception as e:
        print(f"ERROR during AI analysis: {e}")
        return {"error": "Failed to get a valid response from the AI model.", "details": str(e)}

# --- API Routes ---

@app.get("/api/test")
async def test_route():
    """A simple test route to check if the backend is running and if the API key is configured."""
    if IS_GEMINI_CONFIGURED:
        return {"message": "Python backend is alive and Gemini is configured!"}
    else:
        return {"message": "Python backend is alive, but FAILED to configure Gemini API. Check Vercel logs for FATAL_ERROR."}

@app.post("/api/analyze_resume")
async def analyze_resume(file: UploadFile, job_desc: str = Form(...)):
    """The main endpoint to analyze a resume."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save the uploaded file temporarily
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Process the file in a separate thread to avoid blocking
        resume_text = await run_in_threadpool(extract_text_from_pdf, file_path)
        if not resume_text:
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        # Analyze the extracted text
        result = await run_in_threadpool(analyze_resume_with_ai, resume_text, job_desc)
        return result

    except Exception as e:
        # Catch any other unexpected errors
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

    finally:
        # Clean up by deleting the temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

