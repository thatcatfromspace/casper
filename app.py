import streamlit as st
import json
from datetime import datetime, timedelta
import numpy as np
import os

# Import functions from main.py
from main import (
    transcribe,
    parse_command,
    schedule_event,
    save_note,
    save_reminder,
    add_todo,
    start_study_timer,
)

st.set_page_config(page_title="Voice Assistant Dashboard", layout="centered")
st.title("Voice Assistant Dashboard")

st.sidebar.header("Actions")
action = st.sidebar.selectbox(
    "Choose an action",
    ["Schedule Event", "Add Note", "Set Reminder", "Add To-Do", "Start Study Timer"],
)

if action == "Schedule Event":
    subject = st.text_input("Event Subject", "Meeting")
    time = st.text_input("Event Time", "10:00 AM")
    if st.button("Schedule"):
        response = schedule_event(subject, time)
        st.success(response)
        # Show current schedule
        if os.path.exists("schedule.json"):
            with open("schedule.json", "r") as f:
                schedule = json.load(f)
            st.json(schedule)

elif action == "Add Note":
    note = st.text_area("Note Content")
    if st.button("Save Note"):
        response = save_note(note)
        st.success(response)
        # Show notes
        if os.path.exists("notes.json"):
            with open("notes.json", "r") as f:
                notes = json.load(f)
            st.json(notes)

elif action == "Set Reminder":
    reminder = st.text_area("Reminder Content")
    if st.button("Set Reminder"):
        response = save_reminder(reminder)
        st.success(response)
        # Show reminders
        if os.path.exists("reminders.json"):
            with open("reminders.json", "r") as f:
                reminders = json.load(f)
            st.json(reminders)

elif action == "Add To-Do":
    todo = st.text_input("To-Do Item")
    if st.button("Add To-Do"):
        response = add_todo(todo)
        st.success(response)
        # Show todos
        if os.path.exists("todos.json"):
            with open("todos.json", "r") as f:
                todos = json.load(f)
            st.json(todos)

elif action == "Start Study Timer":
    minutes = st.number_input("Minutes", min_value=1, max_value=180, value=25)
    if st.button("Start Timer"):
        st.info(
            f"Timer started for {minutes} minutes. (Voice notification will play at the end if supported)"
        )
        start_study_timer(minutes)

st.sidebar.header("Voice Command (Experimental)")

# --- Voice Recording Widget ---
try:
    from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
    import av
    import io

    class AudioProcessor(AudioProcessorBase):
        def recv(self, frame):
            # Just pass through, we only want the raw audio
            return frame

    ctx = webrtc_streamer(
        key="audio",
        mode="SENDONLY",
        audio_receiver_size=1024,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=AudioProcessor,
        async_processing=True,
    )
    if ctx.audio_receiver:
        audio_frames = ctx.audio_receiver.get_frames(timeout=1)
        audio_bytes = b""
        for audio_frame in audio_frames:
            buf = io.BytesIO()
            audio_frame.to_ndarray().astype(np.int16).tofile(buf)
            audio_bytes += buf.getvalue()
        if audio_bytes:
            st.sidebar.write("Audio recorded. Processing...")
            text = transcribe(audio_bytes)
            st.sidebar.write(f"Recognized: {text}")
            action, arg1, arg2 = parse_command(text)
            if action == "schedule":
                response = schedule_event(arg1, arg2)
                st.sidebar.success(response)
            elif action == "note":
                response = save_note(arg1)
                st.sidebar.success(response)
            elif action == "reminder":
                response = save_reminder(arg1)
                st.sidebar.success(response)
            elif action == "todo":
                response = add_todo(arg1)
                st.sidebar.success(response)
            elif action == "timer":
                st.sidebar.info(f"Timer started for {arg1} minutes.")
                start_study_timer(arg1)
            else:
                st.sidebar.warning("Command not recognized.")
except ImportError:
    st.sidebar.warning(
        "streamlit-webrtc is not installed. Please install it for voice recording."
    )

# --- Fallback: File uploader ---
voice_command = st.sidebar.file_uploader("Upload WAV audio for command", type=["wav"])
if voice_command is not None:
    audio_bytes = voice_command.read()
    text = transcribe(audio_bytes)
    st.sidebar.write(f"Recognized: {text}")
    action, arg1, arg2 = parse_command(text)
    if action == "schedule":
        response = schedule_event(arg1, arg2)
        st.sidebar.success(response)
    elif action == "note":
        response = save_note(arg1)
        st.sidebar.success(response)
    elif action == "reminder":
        response = save_reminder(arg1)
        st.sidebar.success(response)
    elif action == "todo":
        response = add_todo(arg1)
        st.sidebar.success(response)
    elif action == "timer":
        st.sidebar.info(f"Timer started for {arg1} minutes.")
        start_study_timer(arg1)
    else:
        st.sidebar.warning("Command not recognized.")
