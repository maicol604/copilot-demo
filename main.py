import os
import json
import requests
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import google.generativeai as genai
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Coach Trainer AI API")

# --- CORS CONFIGURATION ---
# Allow frontend server (Live Server, etc.) to access the API
origins = [
    "http://127.0.0.1:5500", # Typical VS Code Live Server port
    "http://localhost:5500",
    "http://127.0.0.1:3000", # Alternative port
    "*"                       # Wildcard allows ANY origin (development only)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (POST, GET, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- CONFIGURATION ---
# All credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not SUPABASE_URL:
    print("ERROR: SUPABASE_URL no configurada")
if not SUPABASE_KEY:
    print("ERROR: SUPABASE_KEY no configurada")

# Inicialización segura
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error inicializando Supabase: {e}")

# Validate required environment variables
if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_KEY, GROQ_API_KEY]):
    raise ValueError("Missing required environment variables: SUPABASE_URL, SUPABASE_KEY, GEMINI_KEY, GROQ_API_KEY")

# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-3-flash-preview')
groq_client = Groq(api_key=GROQ_API_KEY)

# --- SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are an expert assistant for sports coaches. Your objective is to process voice transcriptions.
You must extract the client name and generate two versions of the message.

Format instructions:
- 'content_improved': A technical, professional and direct summary for the coach.
- 'content_client_ready': A motivational, empathetic message with marketing style for the client.

IMPORTANT: Respond strictly in plain JSON format, without markdown:
{
  "client_name": "Client Name",
  "content_improved": "Improved text",
  "content_client_ready": "Text for client"
}
"""

@app.post("/process-voice-note")
async def process_voice_note(file: UploadFile = File(...)):
    try:
        # 1. Transcribe with Groq Whisper API
        audio_content = await file.read()
        
        # Create a file-like object for Groq
        from io import BytesIO
        audio_file = BytesIO(audio_content)
        audio_file.name = file.filename
        
        # Call Groq Whisper API
        transcript = groq_client.audio.transcriptions.create(
            file=(file.filename, audio_file, file.content_type),
            model="whisper-large-v3-turbo",
            language="es"  # Spanish language
        )
        original_text = transcript.text

        # 2. FETCH ALL CLIENTS from DB
        clients_res = supabase.table("profiles").select("id, full_name").eq("role", "client").execute()
        clients_list = clients_res.data # List of {"id": "...", "full_name": "..."}

        # 3. GEMINI WITH CONTEXT - Match client and improve content
        prompt = f"""
        TRANSCRIPTION: "{original_text}"
        
        LIST OF REGISTERED CLIENTS:
        {json.dumps(clients_list)}
        
        TASK:
        1. Identify which client from the list is mentioned in the transcription (even if misspelled).
        2. If multiple possible matches (up to 3), return them as candidates. If no match, return empty candidates array.
        3. Improve the note for the coach and create a motivational message for the client.

        RESPONSE FORMAT (JSON):
        {{
          "candidates": [{{ "id": "uuid", "full_name": "Name" }}],
          "content_improved": "Coach summary - professional and technical",
          "content_client_ready": "Motivational message for client with empathy and marketing style"
        }}
        """

        gemini_res = gemini_model.generate_content(prompt)
        # Clean and parse JSON
        cleaned_json = gemini_res.text.replace("```json", "").replace("```", "").strip()
        ai_data = json.loads(cleaned_json)

        return {
            "original_text": original_text,
            "content_improved": ai_data.get("content_improved", ""),
            "content_client_ready": ai_data.get("content_client_ready", ""),
            "candidates": ai_data.get("candidates", [])
        }
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/confirm-and-save")
async def confirm_and_save(
    monitor_id: str = Form(...),
    client_id: str = Form(...),
    content_original: str = Form(...),
    content_improved: str = Form(...),
    content_client_ready: str = Form(...),
    category: Optional[str] = Form(None)
):
    """
    Save the confirmed training note to the database.
    """
    try:
        data = {
            "monitor_id": monitor_id,
            "client_id": client_id,
            "content_original": content_original,
            "content_improved": content_improved,
            "content_client_ready": content_client_ready,
            "status": "accepted",
            "category": category
        }
        
        res = supabase.table("training_notes").insert(data).execute()
        return {"message": "Note saved successfully", "data": res.data}
    
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving: {str(e)}")

@app.get("/get-all-notes")
async def get_all_notes():
    """
    Retrieve all training notes with client information.
    """
    try:
        # Fetch notes with related client profile information
        res = supabase.table("training_notes") \
            .select("*, profiles!training_notes_client_id_fkey(full_name)") \
            .order("created_at", desc=True) \
            .execute()
        
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/get-all-clients")
async def get_all_clients():
    """
    Retrieve all registered clients for manual search.
    """
    try:
        res = supabase.table("profiles") \
            .select("id, full_name") \
            .eq("role", "client") \
            .order("full_name", desc=False) \
            .execute()
        
        return res.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reprocess-text")
async def reprocess_text(request: dict):
    """
    Re-process and improve text using Gemini AI.
    """
    try:
        text = request.get("text", "")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        prompt = f"""
        You are an expert assistant for sports coaches. Improve and polish the following coaching note.
        Make it more professional, technical, and clear while maintaining the original meaning.
        
        ORIGINAL TEXT: "{text}"
        
        Return ONLY the improved text without any JSON format or markdown code blocks.
        """
        
        gemini_res = gemini_model.generate_content(prompt)
        improved_text = gemini_res.text.strip()
        
        return {"improved_text": improved_text}
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update-note")
async def update_note(
    note_id: str = Form(...),
    content_improved: str = Form(...),
    content_client_ready: str = Form(...)
):
    """
    Update an existing training note with new content.
    """
    try:
        data = {
            "content_improved": content_improved,
            "content_client_ready": content_client_ready
        }
        
        res = supabase.table("training_notes") \
            .update(data) \
            .eq("id", note_id) \
            .execute()
        
        return {"message": "Note updated successfully", "data": res.data}
    
    except Exception as e:
        print(f"DEBUG ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8001)) 
    uvicorn.run(app, host="0.0.0.0", port=port)