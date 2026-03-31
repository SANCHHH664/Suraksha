import subprocess
import os
import time
from datetime import datetime

# Fake files for testing
v = "test_vid.avi"
a = "test_aud.wav"
o = "test_out.mp4"

# Create dummy video file using ffmpeg
subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'testsrc=duration=2:size=640x480:rate=20', v], check=True)
# Create dummy audio file using ffmpeg
subprocess.run(['ffmpeg', '-y', '-f', 'lavfi', '-i', 'sine=frequency=440:duration=2', a], check=True)

def mux_av(v, a, o):
    print(f"Starting test muxing for {o}")
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', v, '-i', a, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', o],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        print(f"Muxing complete and saved: {o}")
    except Exception as e:
        print(f"FFmpeg error: {e}")

mux_av(v, a, o)

# Cleanup
if os.path.exists(v): os.remove(v)
if os.path.exists(a): os.remove(a)
if os.path.exists(o): 
    print(f"Success! {o} exists.")
    os.remove(o)
else:
    print("Failure: Output file not created.")
