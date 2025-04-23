import numpy as np

# Whisper pipeline and related imports should be initialized in main.py and passed if needed


def audio_bytes_to_float_array(audio_bytes):
    """Convert raw audio bytes to a normalized float32 numpy array."""
    return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_audio(audio_bytes, whisper_pipeline):
    """Transcribe audio bytes using a provided HuggingFace Whisper pipeline."""
    audio_array = audio_bytes_to_float_array(audio_bytes)
    return whisper_pipeline(audio_array)["text"]
