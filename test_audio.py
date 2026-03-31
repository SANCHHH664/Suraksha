import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

fs = 44100
duration = 2  # seconds

try:
    print("Devices:", sd.query_devices())
    print("Default Input Device:", sd.default.device[0])
    
    print(f"Recording {duration} seconds...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    
    filename = "test_audio_diag.wav"
    wav.write(filename, fs, audio_data)
    print(f"Saved to {filename}")
    if os.path.exists(filename):
        os.remove(filename)
        print("Test file deleted.")

except Exception as e:
    print(f"Error during audio test: {e}")
