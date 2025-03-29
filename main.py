import pyaudio
import numpy as np
from transformers import pipeline, BarkProcessor, BarkModel
import torch
import sounddevice as sd
from datetime import datetime, timedelta
import json

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz

whisper = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    generate_kwargs={"language": "<|en|>"}
)

bark_processor = BarkProcessor.from_pretrained("suno/bark-small")
bark_model = BarkModel.from_pretrained("suno/bark-small").to(device)
print(f"Bark model on: {next(bark_model.parameters()).device}")

def record_audio():
    """Record audio until user stops (Enter key assumed via timeout here)."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording... (Press Ctrl+C to stop or wait 5s)")
    
    frames = []
    try:
        for _ in range(int(RATE / CHUNK * 5)):  # 5-second timeout
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    
    print("Recording stopped.")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b''.join(frames)

def transcribe(audio_data):
    """Convert audio to text with Whisper."""
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return whisper(audio_array)["text"]

def parse_command(text):
    """Simple parsing for 'schedule [subject] [time]'."""
    words = text.lower().split()
    if "schedule" in words:
        subject = words[-2] if len(words) > 2 else "event"  
        time = words[-1] if len(words) > 1 else "unknown"  
        return "schedule", subject, time
    return None, None, None

def schedule_event(subject, time, filename="schedule.json"):
    """Save event to JSON, assuming 'tomorrow' for simplicity."""
    try:
        with open(filename, "r") as f:
            schedule = json.load(f)
    except FileNotFoundError:
        schedule = {}
    
    date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    schedule[date] = {subject: time}
    with open(filename, "w") as f:
        json.dump(schedule, f)
    return f"Scheduled {subject} for {time} tomorrow."

def speak(text):
    """Generate and play speech with Bark on CUDA."""
    inputs = bark_processor(text, voice_preset="v2/en_speaker_3")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Ensure all tensors are on CUDA
    with torch.no_grad():
        audio = bark_model.generate(**inputs).cpu().numpy().squeeze()
    sd.play(audio, samplerate=bark_model.generation_config.sample_rate)
    sd.wait()

try:
    audio = record_audio()
    text = transcribe(audio)
    print(f"Heard: {text}")
    
    action, subject, time = parse_command(text)
    if action == "schedule":
        response = schedule_event(subject, time)
        print(response)
        speak(response)
    else:
        speak("I only know how to schedule things right now!")
except Exception as e:
    print(f"Error: {e}")
    speak("Oops, something went wrong!")