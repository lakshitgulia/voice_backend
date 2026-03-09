
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
import shutil, os, joblib, cv2
import numpy as np
import os
from gtts import gTTS
import re
import requests

app = FastAPI()


HF_TOKEN = os.getenv("HF_TOKEN")

def speech_to_text(file_path):

    API_URL = "https://api-inference.huggingface.co/models/openai/whisper-small"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }

    with open(file_path, "rb") as f:
        response = requests.post(API_URL, headers=headers, data=f)

    result = response.json()

    return result["text"]

# ================= PATHS =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD = os.path.join(BASE_DIR, "uploads")
AUDIO = os.path.join(BASE_DIR, "audio")

os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(AUDIO, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO), name="audio")

# ================= GLOBAL MODELS =================

intent_model = None
vectorizer = None
symbol_model = None
class_names = None

# ================= COMPLETE WARNING DATABASE =================

warning_database = {

    "check_engine": {
        "problem": "Engine control system malfunction detected.",
        "solution": "Possible sensor, fuel injection or emission issue. Avoid aggressive driving and schedule diagnostics immediately.",
        "severity": "medium"
    },

    "engine_temp": {
        "problem": "Engine overheating detected.",
        "solution": "Stop vehicle immediately. Turn off AC. Allow engine to cool. Do NOT open radiator cap while hot.",
        "severity": "high"
    },

    "oil_pressure": {
        "problem": "Low engine oil pressure.",
        "solution": "Stop driving immediately. Check oil level. Continuing may cause severe engine damage.",
        "severity": "high"
    },

    "coolant_low": {
        "problem": "Coolant level low.",
        "solution": "Allow engine to cool and refill coolant. Check for leaks.",
        "severity": "high"
    },

    "transmission": {
        "problem": "Transmission system fault.",
        "solution": "Gear shifting may become rough. Avoid long drives and visit service center.",
        "severity": "high"
    },

    "abs": {
        "problem": "ABS malfunction.",
        "solution": "Braking works but anti-lock braking is disabled. Drive cautiously.",
        "severity": "medium"
    },

    "brake_warning": {
        "problem": "Brake system issue detected.",
        "solution": "Possible brake fluid leak or brake failure. Do not drive if pedal feels soft.",
        "severity": "high"
    },

    "regenerative_brake_fault": {
        "problem": "Regenerative braking unavailable.",
        "solution": "Vehicle will rely on mechanical brakes. Service recommended.",
        "severity": "medium"
    },

    "airbag": {
        "problem": "Airbag system malfunction.",
        "solution": "Airbags may not deploy during collision. Immediate inspection required.",
        "severity": "high"
    },

    "power_steering": {
        "problem": "Power steering malfunction.",
        "solution": "Steering may become heavy. Service immediately.",
        "severity": "high"
    },

    "tpms": {
        "problem": "Tyre pressure low.",
        "solution": "Inflate tyres to recommended PSI. Inspect for punctures.",
        "severity": "medium"
    },

    "battery": {
        "problem": "Battery charging problem.",
        "solution": "Possible alternator failure or weak battery. Check terminals.",
        "severity": "medium"
    },

    "ev_battery_fault": {
        "problem": "High-voltage EV battery fault.",
        "solution": "Stop vehicle safely and contact authorized EV service center.",
        "severity": "high"
    },

    "charging_fault": {
        "problem": "Charging system error.",
        "solution": "Check charging cable and power source.",
        "severity": "medium"
    },

    "hybrid_system_fault": {
        "problem": "Hybrid system malfunction.",
        "solution": "Visit service center immediately.",
        "severity": "high"
    },

    "emission_system": {
        "problem": "Emission control system fault.",
        "solution": "Schedule inspection soon.",
        "severity": "medium"
    },

    "catalytic_converter": {
        "problem": "Catalytic converter overheating.",
        "solution": "Reduce speed immediately and service urgently.",
        "severity": "high"
    },

    "lane_departure": {
        "problem": "Lane assist malfunction.",
        "solution": "Clean windshield camera area.",
        "severity": "medium"
    },

    "blind_spot": {
        "problem": "Blind spot monitoring fault.",
        "solution": "Sensor obstruction possible.",
        "severity": "medium"
    },

    "forward_collision": {
        "problem": "Forward collision warning unavailable.",
        "solution": "Front radar may be blocked.",
        "severity": "high"
    },

    "adaptive_cruise": {
        "problem": "Adaptive cruise control malfunction.",
        "solution": "Radar sensor may be blocked.",
        "severity": "medium"
    },

    "driver_attention": {
        "problem": "Driver fatigue detected.",
        "solution": "Take a break before continuing.",
        "severity": "low"
    },

    "immobilizer": {
        "problem": "Key not detected.",
        "solution": "Check key battery or try spare key.",
        "severity": "medium"
    },

    "service_due": {
        "problem": "Scheduled service due.",
        "solution": "Routine maintenance required.",
        "severity": "low"
    },

    "fuel_low": {
        "problem": "Fuel level low.",
        "solution": "Refuel at nearest fuel station.",
        "severity": "low"
    }
}
intent_keywords = {

    "engine_temp": ["overheat","overheating","heated","hot","engine hot","temperature","high temp"],
    "oil_pressure": ["oil pressure","engine oil","low oil"],
    "coolant_low": ["coolant","radiator"],
    "transmission": ["gear","gearbox","transmission"],
    "abs": ["abs"],
    "brake_warning": ["brake"],
    "regenerative_brake_fault": ["regenerative"],
    "airbag": ["airbag"],
    "power_steering": ["steering"],
    "tpms": ["tyre","tire","pressure","puncture"],
    "battery": ["battery","alternator"],
    "ev_battery_fault": ["high voltage"],
    "charging_fault": ["charging"],
    "hybrid_system_fault": ["hybrid"],
    "emission_system": ["emission"],
    "catalytic_converter": ["catalytic"],
    "lane_departure": ["lane"],
    "blind_spot": ["blind spot"],
    "forward_collision": ["collision","radar"],
    "adaptive_cruise": ["cruise"],
    "driver_attention": ["fatigue","sleepy"],
    "immobilizer": ["key not detected"],
    "service_due": ["service"],
    "fuel_low": ["fuel","petrol","diesel"],
    "check_engine": ["engine light","check engine"]
}
# ================= STARTUP =================

@app.on_event("startup")
def load_models():

    global intent_model, vectorizer, symbol_model, class_names

    try:
        intent_model, vectorizer = joblib.load("models/reply_model.pkl")
    except:
        print("ML model not found → keyword matching used")

    model_path = os.path.join(BASE_DIR, "models", "dashboard_symbol_model_fixed.keras")

    symbol_model = tf.keras.models.load_model(model_path, compile=False)

    class_path = os.path.join(BASE_DIR, "models", "class_names.json")
    with open(class_path, "r") as f:
        class_names = json.load(f)

# ================= TEXT CLEANING =================

def preprocess(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z ]',' ',text)

    text = re.sub(r'\s+',' ',text)

    return text.strip()

# ================= INTENT DETECTION =================

def detect_intent(text):

    text = preprocess(text)

    if intent_model and vectorizer:

        vec = vectorizer.transform([text])

        pred = intent_model.predict(vec)[0]

        if pred in warning_database:
            return pred

    scores = {}

    for intent,keywords in intent_keywords.items():

        for word in keywords:

            if word in text:

                scores[intent] = scores.get(intent,0)+1

    if scores:

        return max(scores,key=scores.get)

    return None


# ================= COLOR DETECTION =================

def detect_color(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return "unknown"

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    red = cv2.inRange(hsv,np.array([0,120,70]),np.array([10,255,255]))

    yellow = cv2.inRange(hsv,np.array([20,100,100]),np.array([30,255,255]))

    if np.sum(red)>2000:
        return "red"

    if np.sum(yellow)>2000:
        return "yellow"

    return "unknown"


# ================= SYMBOL DETECTION =================

def detect_symbol(image_path):

    img = image.load_img(image_path, target_size=(128,128))

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    preds = symbol_model.predict(img_array)

    idx = np.argmax(preds)

    symbol = class_names[str(idx)]   

    return symbol


# ================= TEXT API =================

@app.post("/send_text")

def send_text(message: str = Form(...)):

    intent_key = detect_intent(message)

    if intent_key is None:

        reply = "I couldn't clearly understand the issue. Please describe the problem in more detail."

        audio_path = os.path.join(AUDIO,"reply.mp3")

        tts = gTTS(text=reply)
        tts.save(audio_path)

        return {

            "detected_intent":None,
            "ai_response_text":reply,
            "audio_url":"/audio/reply.mp3",
            "book_service_prompt":False
        }

    data = warning_database[intent_key]

    reply = f"{data['problem']} Solution: {data['solution']}"

    book = data["severity"]=="high"

    audio_path = os.path.join(AUDIO,"reply.wav")

    tts = gTTS(text=reply)
    tts.save(audio_path)

    return {

        "detected_intent":intent_key,
        "ai_response_text":reply,
        "audio_url":"/audio/reply.wav",
        "book_service_prompt":book
    }


# ================= AUDIO API =================

@app.post("/send_audio")

async def send_audio(file: UploadFile = File(...)):

    path = os.path.join(UPLOAD,"input.wav")

    with open(path,"wb") as buffer:

        shutil.copyfileobj(file.file,buffer)

    text = speech_to_text(path)

    intent_key = detect_intent(text)

    if intent_key is None:

        reply = "I couldn't clearly understand the issue. Please describe the problem in more detail."

        audio_path = os.path.join(AUDIO,"reply.wav")

        tts = gTTS(text=reply)
        tts.save(audio_path)  

        return {

            "transcription":text,
            "detected_intent":None,
            "ai_response_text":reply,
            "audio_url":"/audio/reply.wav"
        }

    data = warning_database[intent_key]

    reply = f"{data['problem']} Solution: {data['solution']}"

    audio_path = os.path.join(AUDIO,"reply.wav")

    tts = gTTS(text=reply)
    tts.save(audio_path)

    return {

        "transcription":text,
        "detected_intent":intent_key,
        "ai_response_text":reply,
        "audio_url":"/audio/reply.wav"
    }


# ================= IMAGE API =================

@app.post("/upload_warning_light")

async def upload_warning_light(file: UploadFile = File(...)):

    file_path = f"uploads/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    symbol = detect_symbol(file_path)
    color = detect_color(file_path)

    if symbol not in warning_database:

        return {
            "symbol_detected": symbol,
            "color_detected": color,
            "problem": "Symbol detected but not mapped in database.",
            "solution": "Please update warning database."
        }

    warning = warning_database[symbol]

    return {
        "symbol_detected": symbol,
        "color_detected": color,
        "problem": warning["problem"],
        "solution": warning["solution"]
    }


# ================= RUN =================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
