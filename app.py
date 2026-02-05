from fastapi import FastAPI, Header, HTTPException, Form
import tensorflow as tf
import numpy as np
import librosa
import base64
import os

MODEL_PATH = "model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

THRESHOLD = 0.9
API_KEY = "12345"

app = FastAPI(title="AI Voice Detection API")

@app.post("/predict")
def predict(
    language: str = Form(...),
    audio_format: str = Form(...),
    audio_base64: str = Form(...),
    x_api_key: str = Header(...)
):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        audio_bytes = base64.b64decode(audio_base64)

        temp_file = "temp.wav"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)

        audio, sr = librosa.load(temp_file, sr=16000)

        if len(audio) > 16000:
            audio = audio[:16000]
        else:
            audio = np.pad(audio, (0, 16000 - len(audio)))

        audio_input = np.expand_dims(audio, axis=0)

        prob = model.predict(audio_input)[0][0]
        label = "AI" if prob > THRESHOLD else "Human"

        return {
            "prediction": label,
            "ai_probability": float(prob),
            "human_confidence": float(1 - prob)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
