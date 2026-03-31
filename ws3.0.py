from flask import Flask, Response, jsonify, render_template
from flask_cors import CORS
import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import threading
import time
import os
import webbrowser
import urllib.request
import json
import subprocess
import sounddevice as sd
import scipy.io.wavfile as wav

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
VIDEO_DIR = os.path.join(BASE_DIR, "captures")


if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

urls = {
    "gender_deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/gender_deploy.prototxt",
    "gender_net.caffemodel":  "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel"
}
for filename, url in urls.items():
    filepath = os.path.join(BASE_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"{filename} downloaded.")
        except Exception as e:
            print(f"Download failed for {filename}: {e}")

# Models & AI
gender_net  = None
gender_list = ['Male', 'Female']
try:
    proto  = os.path.join(BASE_DIR, "gender_deploy.prototxt")
    model_w = os.path.join(BASE_DIR, "gender_net.caffemodel")
    gender_net = cv2.dnn.readNet(model_w, proto)
    print("Gender AI Loaded Successfully")
except Exception as e:
    print(f"Error loading Gender Model: {e}")

try:
    yolo_model = YOLO("yolov8n.pt")
    print("YOLO Loaded Successfully")
except Exception as e:
    yolo_model = None
    print(f"YOLO load failed: {e}")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

live_location = "Location: Unknown"
try:
    with urllib.request.urlopen("http://ip-api.com/json/", timeout=5) as response:
        data = json.loads(response.read().decode())
        if data.get("status") == "success":
            live_location = f"{data['lat']} N, {data['lon']} E ({data['city']})"
            print(f"Live Location acquired: {live_location}")
except Exception as e:
    print(f"Location fetch failed: {e}")

current_frame = None
frame_lock    = threading.Lock()

stats = {"males": 0, "females": 0, "total": 0, "risk": 0, "alert": "Safe"}

alerts_queue = []
alerts_lock  = threading.Lock()

tracker_ids = {}
next_id     = 0
recording   = False
recorder    = None
start_time  = None
cap         = None
current_files = {}
is_monitoring = False

def audio_capture_loop():
    global recording, current_files
    fs = 44100
    while True:
        if recording and "aud" in current_files:
            target_aud = current_files["aud"]
            try:
                # Capture ~20.5 seconds
                audio_data = sd.rec(int(20.5 * fs), samplerate=fs, channels=1, dtype='int16')
                sd.wait()
                wav.write(target_aud, fs, audio_data)
            except Exception as e:
                print("Audio Thread Error:", e)
                time.sleep(1)
            time.sleep(1)
        else:
            time.sleep(0.5)

threading.Thread(target=audio_capture_loop, daemon=True).start()

MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Detection
def classify_gender(crop):
    if gender_net is not None:
        try:
            resized = cv2.resize(crop, (227, 227))
            blob    = cv2.dnn.blobFromImage(resized, 1.0, (227, 227), MEAN_VALUES, swapRB=False)
            gender_net.setInput(blob)
            preds   = gender_net.forward()
            return gender_list[preds[0].argmax()]
        except Exception as e:
            print(f"Gender net error: {e}")

    # Fallback heuristic
    try:
        h, w, _ = crop.shape
        return "Female" if (w / h) > 0.42 else "Male"
    except:
        return "Female"

# Object Tracking
def update_tracker(detections):
    global next_id, tracker_ids
    new_tracker = {}
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cx, cy   = (x1 + x2) // 2, (y1 + y2) // 2
        best_id  = None
        min_dist = float("inf")
        for tid, (px, py) in tracker_ids.items():
            dist = np.hypot(cx - px, cy - py)
            if dist < 60 and dist < min_dist:
                best_id  = tid
                min_dist = dist
        if best_id is None:
            best_id = next_id
            next_id += 1
        new_tracker[best_id] = (cx, cy)
        det["id"] = best_id
    tracker_ids = new_tracker

def push_alert(msg, level="info"):
    global is_monitoring
    if not is_monitoring:
        return
    with alerts_lock:
        alerts_queue.append({"msg": msg, "level": level})

def process_frame(frame):
    global stats, recording, recorder, start_time, current_files, is_monitoring

    h, w, _ = frame.shape
    detections = []
    male, female = 0, 0

    # Scanning effect
    scan_y = int((time.time() * 150) % h)
    cv2.line(frame, (0, scan_y), (w, scan_y), (0, 255, 0), 1)

    # YOLO person detection
    if yolo_model:
        try:
            results = yolo_model(frame, conf=0.5, verbose=False)
            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                        if crop.size > 0:
                            gender = classify_gender(crop)
                            if gender == "Male":
                                male += 1
                            else:
                                female += 1
                            detections.append({"box": (x1, y1, x2, y2), "gender": gender})
        except Exception as e:
            print(f"Detection Error: {e}")

    update_tracker(detections)

    # Risk logic
    total_count = male + female
    prev_risk   = stats.get("risk", 0)
    risk        = 10
    alert_msg   = "Status: Clear"

    if female >= 1:
        if male >= 3:
            risk      = 95
            alert_msg = "HIGH RISK: Potential Harassment"
        elif male >= 2:
            risk      = 70
            alert_msg = "CAUTION: Outnumbered"
        elif male == 1:
            risk      = 40
            alert_msg = "Monitoring: Neutral"
        else:
            risk      = 15
            alert_msg = "Status: Safe"

    # Fire alerts only on level transitions so log doesn't spam
    if risk >= 70 and prev_risk < 70:
        push_alert(alert_msg, "critical")
    elif risk >= 40 and prev_risk < 40:
        push_alert(alert_msg, "warn")
    elif risk < 40 and prev_risk >= 40:
        push_alert("Situation normalised — risk cleared", "info")

    stats = {
        "males":   male,
        "females": female,
        "total":   total_count,
        "risk":    risk,
        "alert":   alert_msg
    }

    # Recording Management
    try:
        if is_monitoring and risk >= 70 and not recording:
            recording = True
            ts       = datetime.now().strftime('%Y%m%d_%H%M%S')
            vid_file = os.path.join(VIDEO_DIR, f"alert_{ts}.avi")
            aud_file = os.path.join(VIDEO_DIR, f"alert_{ts}.wav")
            out_file = os.path.join(VIDEO_DIR, f"alert_{ts}.mp4")
            
            current_files = {"vid": vid_file, "aud": aud_file, "out": out_file}
            
            fourcc   = cv2.VideoWriter_fourcc(*'XVID')
            recorder = cv2.VideoWriter(vid_file, fourcc, 10.0, (w, h)) # Lowered to 10 FPS for normal speed
            start_time = time.time()
            push_alert(f"Recording AV started: alert_{ts}", "critical")
            print(f"Recording Video: {vid_file}")

        if recording and recorder:
            rec_frame = frame.copy()
            
            ts_str  = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            loc_str = f"LOC: {live_location} | TS: {ts_str}"
            
            cv2.putText(rec_frame, loc_str, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(rec_frame, loc_str, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            
            recorder.write(rec_frame)
            if time.time() - start_time > 20:
                recorder.release()
                recorder    = None
                recording   = False
                start_time  = None
                
                vid = current_files.get("vid")
                aud = current_files.get("aud")
                out = current_files.get("out")
                current_files = {}
                
                push_alert("Recording complete. Muxing AV...", "info")
                print("Video saved. Muxing via FFmpeg...")
                
                # Background Muxing
                def mux_av(v, a, o):
                    log_file = os.path.join(VIDEO_DIR, "recording_log.txt")
                    def log(msg):
                        with open(log_file, "a") as f:
                            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
                    
                    log(f"Starting muxing for {o}")
                    # Wait extra for audio thread to flush
                    time.sleep(5.0) 
                    
                    if not os.path.exists(v):
                        log(f"Error: Video file {v} not found.")
                        return
                    if not os.path.exists(a):
                        log(f"Error: Audio file {a} not found.")
                        # Still try to convert video if audio is missing
                        try:
                            subprocess.run(
                                ['ffmpeg', '-y', '-i', v, '-c:v', 'libx264', '-preset', 'ultrafast', o],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                            )
                            if os.path.exists(v): os.remove(v)
                            log(f"Muxing complete (Video only): {o}")
                            push_alert("Video saved (No audio detected)", "warn")
                        except Exception as e:
                            log(f"FFmpeg (Video only) error: {e}")
                        return

                    try:
                        # Re-encoding to H.264/AAC for maximum compatibility
                        subprocess.run(
                            ['ffmpeg', '-y', '-i', v, '-i', a, '-c:v', 'libx264', '-preset', 'ultrafast', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', o],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
                        )
                        if os.path.exists(v): os.remove(v)
                        if os.path.exists(a): os.remove(a)
                        log(f"Muxing complete and saved: {o}")
                        push_alert("AV Recording saved successfully!", "info")
                    except Exception as e:
                        log(f"FFmpeg error: {e}")
                        
                threading.Thread(target=mux_av, args=(vid, aud, out), daemon=True).start()
    except Exception as e:
        print(f"Recording Loop Error: {e}")
        if recorder:
            recorder.release()
            recorder  = None
        recording = False

    for d in detections:
        x1, y1, x2, y2 = d["box"]
        color = (255, 80, 80) if d["gender"] == "Male" else (255, 105, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{d['gender']} #{d.get('id', '?')}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # Overlay text
    if risk >= 70:
        cv2.rectangle(frame, (0, 0), (w, 48), (0, 0, 200), -1)
        cv2.putText(frame, f"!!! {alert_msg} !!!",
                    (w // 10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, alert_msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 80), 2)

    return frame

def video_loop():
    global current_frame, cap
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    time.sleep(2)
    if not cap.isOpened():
        print("Camera not found!")
        return
    print("Camera opened")
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        frame     = cv2.resize(frame, (640, 480))
        try:
            processed = process_frame(frame)
            with frame_lock:
                current_frame = processed.copy()
        except Exception as e:
            print(f"Video Loop Error (Frame dropped): {e}")
            time.sleep(0.01)

threading.Thread(target=video_loop, daemon=True).start()

@app.route('/')
def root():
    return render_template('suraksha.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        global current_frame
        while True:
            with frame_lock:
                frame = current_frame
            if frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def get_stats():
    return jsonify({
        "males":   int(stats.get("males",   0)),
        "females": int(stats.get("females", 0)),
        "total":   int(stats.get("males",   0)) + int(stats.get("females", 0)),
        "cumulative": next_id,
        "risk":    int(stats.get("risk",    0)),
        "alert":   str(stats.get("alert",  "Safe")),
        "location": live_location
    })

@app.route('/alerts')
def get_alerts():
    with alerts_lock:
        pending = alerts_queue.copy()
        alerts_queue.clear()
    return jsonify({"alerts": pending})

@app.route('/api/start', methods=['POST'])
def api_start():
    global is_monitoring
    is_monitoring = True
    return jsonify({"status": "started"})

@app.route('/api/stop', methods=['POST'])
def api_stop():
    global is_monitoring
    is_monitoring = False
    return jsonify({"status": "stopped"})

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
