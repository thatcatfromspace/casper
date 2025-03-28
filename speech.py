import pyaudio
import keyboard
from transformers import pipeline, BarkProcessor, BarkModel
import spacy
import numpy as np
import json
from datetime import datetime, timedelta
import sounddevice as sd

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    generate_kwargs={"language": "en", "task": "translate"} 
)
spacey_model = spacy.load("en_core_web_sm")

bark_processor = BarkProcessor.from_pretrained("suno/bark-small")
bark_model = BarkModel.from_pretrained("suno/bark-small")

# Audio Settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# press Enter to stop recording. 
# TODO: implement better logic and stoppage mechanism using something like webrtcdav
def record_live_audio():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Listening... (Press Enter to stop)")
    
    frames = []
    while not keyboard.is_pressed("enter"):
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    
    print("Stopped recording.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

def transcribe_audio(audio_data):
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    # !important expects audio as a numpy array at the correct sample rate
    transcription = asr_pipeline(audio_array)["text"]
    return transcription

# very naive way to parse commands, but works for now
def parse_command(text):
    doc = spacey_model(text)
    action = None
    subject = None
    time = None
    for token in doc:
        if token.lemma_ in ["schedule", "plan"]:
            action = "schedule"
        elif token.pos_ == "NOUN":
            subject = token.text
    for ent in doc.ents:
        if ent.label_ == "TIME" or ent.label_ == "DATE":
            time = ent.text
    return action, subject, time

def process_schedule(subject, time, schedule_file="schedule.json"):
    try:
        with open(schedule_file, "r") as f:
            schedule = json.load(f)
    except FileNotFoundError:
        schedule = {}
    if "tomorrow" in time.lower():
        date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        schedule[date] = {subject: time}
    with open(schedule_file, "w") as f:
        json.dump(schedule, f)
    return f"{subject} scheduled for {time}."

def speak(text):
    inputs = bark_processor(text, voice_preset="v2/en_speaker_3")
    speech_output = bark_model.generate(**inputs).cpu().numpy().squeeze()
    sampling_rate = bark_model.generation_config.sample_rate  # 24kHz for Bark
    sd.play(speech_output, samplerate=sampling_rate)
    sd.wait()

audio_data = record_live_audio()
text = transcribe_audio(audio_data)
print(f"Transcribed: {text}")
action, subject, time = parse_command(text)
if action == "schedule":
    response = process_schedule(subject, time)
    print(response)
    speak(response)
else:
    speak("Sorry, I only handle scheduling for now!")