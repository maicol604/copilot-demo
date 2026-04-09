from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import os
import shutil

app = FastAPI(title="Whisper Local API")

# Load the model on startup (the 'base' model is a good balance)
# Options: 'tiny', 'base', 'small', 'medium', 'large'
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    # Validate that it's an audio file (optional but recommended)
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.ogg')):
        raise HTTPException(status_code=400, detail="Audio format not supported.")

    temp_file = f"temp_{file.filename}"
    
    try:
        # Save file temporarily
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Transcribe
        result = model.transcribe(temp_file, fp16=False)
        
        return {
            "filename": file.filename,
            "language": result.get("language"),
            "text": result.get("text")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)