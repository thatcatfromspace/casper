import pyaudio
import numpy as np
from transformers import pipeline, BarkProcessor, BarkModel
import torch
import sounddevice as sd
from datetime import datetime, timedelta
import json
import colorama
from colorama import Fore, Style
import threading
import keyboard

colorama.init()

device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"{Fore.GREEN}Device: {device}{Style.RESET_ALL}")

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper expects 16kHz

whisper = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    generate_kwargs={"language": "<|en|>"},
)

bark_processor = BarkProcessor.from_pretrained("suno/bark-small")
bark_model = BarkModel.from_pretrained("suno/bark-small").to(device)
# print(
#     f"{Fore.GREEN}Bark model on: {next(bark_model.parameters()).device}{Style.RESET_ALL}"
# )


def record_audio():
    """Record audio until user stops (Enter key)."""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    print(f"{Fore.BLUE}Recording... (Press Enter to stop){Style.RESET_ALL}")

    frames = []
    recording = True

    def stop_recording():
        nonlocal recording
        keyboard.wait("enter")
        recording = False

    # Start a thread to listen for the Enter key press
    threading.Thread(target=stop_recording, daemon=True).start()

    try:
        while recording:
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    print(f"{Fore.YELLOW}Recording stopped.{Style.RESET_ALL}")
    stream.stop_stream()
    stream.close()
    p.terminate()
    return b"".join(frames)


def transcribe(audio_data):
    """Convert audio to text with Whisper."""
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    return whisper(audio_array)["text"]


def parse_command(text):
    """Parse commands for schedule, note, reminder, todo, and timer."""
    words = text.lower().split()
    if "schedule" in words:
        subject = words[-2] if len(words) > 2 else "event"
        time = words[-1] if len(words) > 1 else "unknown"
        return "schedule", subject, time
    if "note" in words:
        idx = words.index("note")
        note_content = " ".join(words[idx + 1 :]) if idx + 1 < len(words) else ""
        return "note", note_content, None
    if "remind" in words or "reminder" in words:
        idx = words.index("remind") if "remind" in words else words.index("reminder")
        reminder_content = " ".join(words[idx + 1 :]) if idx + 1 < len(words) else ""
        return "reminder", reminder_content, None
    if "todo" in words or "to-do" in words:
        idx = words.index("todo") if "todo" in words else words.index("to-do")
        todo_content = " ".join(words[idx + 1 :]) if idx + 1 < len(words) else ""
        return "todo", todo_content, None
    if "timer" in words or "study" in words:
        # e.g., "start timer 25" or "study for 30 minutes"
        for i, w in enumerate(words):
            if w.isdigit():
                return "timer", int(w), None
        return "timer", 25, None  # default 25 min
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


def save_note(note, filename="notes.json"):
    try:
        with open(filename, "r") as f:
            notes = json.load(f)
    except FileNotFoundError:
        notes = []
    notes.append({"timestamp": datetime.now().isoformat(), "note": note})
    with open(filename, "w") as f:
        json.dump(notes, f)
    return "Note saved."


def save_reminder(reminder, filename="reminders.json"):
    try:
        with open(filename, "r") as f:
            reminders = json.load(f)
    except FileNotFoundError:
        reminders = []
    reminders.append({"timestamp": datetime.now().isoformat(), "reminder": reminder})
    with open(filename, "w") as f:
        json.dump(reminders, f)
    return "Reminder set."


def add_todo(todo, filename="todos.json"):
    try:
        with open(filename, "r") as f:
            todos = json.load(f)
    except FileNotFoundError:
        todos = []
    todos.append({"timestamp": datetime.now().isoformat(), "todo": todo, "done": False})
    with open(filename, "w") as f:
        json.dump(todos, f)
    return "To-do added."


def start_study_timer(minutes):
    import time

    speak(f"Starting a {minutes} minute study timer.")
    time.sleep(minutes * 60)
    speak("Time's up! Take a break.")


try:
    audio = record_audio()
    text = transcribe(audio)
    print(f"{Fore.CYAN}Heard: {text}{Style.RESET_ALL}")
    action, arg1, arg2 = parse_command(text)
    if action == "schedule":
        response = schedule_event(arg1, arg2)
        print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
        speak(response)
    elif action == "note":
        response = save_note(arg1)
        print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
        speak(response)
    elif action == "reminder":
        response = save_reminder(arg1)
        print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
        speak(response)
    elif action == "todo":
        response = add_todo(arg1)
        print(f"{Fore.GREEN}{response}{Style.RESET_ALL}")
        speak(response)
    elif action == "timer":
        start_study_timer(arg1)
    else:
        speak(
            "I can schedule, take notes, set reminders, manage to-dos, and start a study timer!"
        )
except Exception as e:
    print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
    speak("Oops, something went wrong!")
