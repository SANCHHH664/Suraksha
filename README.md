# Suraksha - AI-Powered Women's Safety Monitoring System

**Suraksha** is a real-time, AI-driven monitoring system designed to enhance safety for women in public and private spaces. Using computer vision and machine learning, it identifies high-risk situations (e.g., being outnumbered by males) and automatically triggers audio-video recording for evidence collection.

## 🚀 Features
- **Real-time AI Detection**: Uses YOLOv8 for person detection and Caffe-based DNN for gender classification.
- **Dynamic Risk Assessment**: Continuously monitors the ratio of males to females to calculate risk levels.
- **Automated Evidence Collection**: Automatically records synchronized audio and video (`.mp4`) during high-risk scenarios.
- **Interactive Dashboard**: A clean, responsive web interface for live monitoring, alerts, and system control.
- **Live Location Tracking**: Automatically fetches current device location for better context during emergency alerts.

## 🛠️ Technology Stack
- **Backend**: Python, Flask, Flask-CORS
- **Computer Vision**: OpenCV, YOLOv8 (Ultralytics)
- **Deep Learning**: Caffe (Gender Detection)
- **Audio Processing**: SoundDevice, Scipy
- **Frontend**: HTML5, CSS3, JavaScript

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/suraksha.git
   cd suraksha
   ```

2. **Install dependencies**:
   Ensure you have FFmpeg installed on your system (required for video/audio muxing).
   ```bash
   pip install -r requirements.txt
   ```

3. **Required Hardware**:
   - Web camera (for video analytics)
   - Microphone (for audio recording)

## 🚦 Usage

1. **Start the application**:
   ```bash
   python womensafety.py
   ```
   The browser will automatically open `http://127.0.0.1:5000/`.

2. **Controls**:
   - **Start Monitoring**: Enables AI detection and risk assessment.
   - **Video Feed**: View live camera stream with bounding boxes and gender labels.
   - **Risk Dashboard**: Monitor real-time statistics (Male/Female count) and risk level.

## 📁 Directory Structure
- `app.py`: Main Flask server and AI logic.
- `templates/`: HTML frontend templates.
- `static/`: Frontend assets (CSS, JS).
- `models/`: (Optional) AI models; downloaded automatically if missing.
- `captures/`: Directory where alert recordings are stored (Git-ignored).

## 🛡️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
