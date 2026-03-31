from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import threading

app = Flask(__name__)
CORS(app)

# Global variables
current_frame = None
frame_lock = threading.Lock()

# Video capture thread
def video_capture():
    global current_frame
    cap = cv2.VideoCapture(0)  # Webcam
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 480))
        
        with frame_lock:
            current_frame = frame.copy()
    
    cap.release()

# Start video capture in background
video_thread = threading.Thread(target=video_capture, daemon=True)
video_thread.start()

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if current_frame is not None:
                    _, buffer = cv2.imencode('.jpg', current_frame)
                    frame_data = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_data)).encode() + b'\r\n\r\n'
                   + frame_data + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Live Camera Feed</title>
    </head>
    <body>
        <h1>Live Camera Feed</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='127.0.0.1', port=5001)