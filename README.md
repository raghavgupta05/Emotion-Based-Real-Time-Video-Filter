# 🎭 Emotion-Based Real-Time Video Filter

![Status](https://img.shields.io/badge/Project%20Status-Completed-brightgreen)
![Python](https://img.shields.io/badge/Python-3.7+-blue)
![OpenCV](https://img.shields.io/badge/Framework-OpenCV-green)
![Deep Learning](https://img.shields.io/badge/AI-FER%20%2B%20MTCNN-orange)

---

## 📌 Project Overview

A real-time emotion detection system that applies dynamic visual filters to webcam feed based on facial expressions.  
This project uses **Deep Learning (FER + MTCNN)** to detect emotions and overlay atmospheric effects.

✔ Real-time webcam processing  
✔ 4 emotion-based filters — **Happy, Sad, Surprise, Angry**  
✔ Implemented using **OpenCV + TensorFlow**

---

## 📂 Repository Structure

```
├📂 emotion-video-filter/
│
├── 📄 final.py                          # Main application (4 emotions)
├── 📄 V3.py                             # Enhanced version (3 emotions)
├── 📄 V1.py                             # Basic version (happy/sad)
├── 📄 Hybrid.py                         # Image integration test
├── 📄 surprise.py                       # Surprise filter prototype
│
├── 📁 assets/
│   ├── sun.png                          # Sunshine overlay
│   ├── cloud.png                        # Cloud overlay
│   └── fumes.png                        # Fumes overlay
│
├── 📄 requirements.txt
└── 📄 README.md
```

---

## 🎨 Filter Effects

| Emotion | Effect | Description |
|---------|--------|-------------|
| 😊 **Happy** | ☀️ Sunshine | Yellow overlay + animated sun |
| 😢 **Sad** | 🌧️ Rain | Blue tint + falling raindrops |
| 😮 **Surprise** | ⛈️ Storm | Cloud + lightning bolts |
| 😠 **Angry** | 💨 Fumes | Red tint + rising steam |

---

## 🔧 Tools & Technologies

- Python 3.7+
- OpenCV (cv2)
- FER (Facial Emotion Recognition)
- TensorFlow
- NumPy
- MTCNN (Face Detection)

---

## 🧠 Model Architecture

- **Face Detection** → Haar Cascade + MTCNN
- **Emotion Analysis** → Pre-trained CNN (FER library)
- **Emotion Smoothing** → Rolling average (10 frames)
- **Visual Effects** → Alpha blending + particle system

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/raghavgupta05/emotion-video-filter.git
cd emotion-video-filter
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Image Assets (Optional)
Place in project directory:
- sun.png
- cloud.png
- fumes.png

*System uses fallback graphics if not provided*

### 4. Run the Application
```bash
python final.py
```

**Press 'q' to quit**

---

## 📈 System Performance

| Metric | Value |
|--------|-------|
| Frame Rate | 10-20 FPS (CPU) |
| Detection Accuracy | ~85% (FER + MTCNN) |
| Latency | <100ms per frame |
| Smoothing Window | 10 frames |

---

## 📉 Key Features

✔ Real-time emotion detection  
✔ Smooth filter transitions  
✔ Alpha-blended overlays  
✔ Particle-based rain animation  
✔ Multi-face support  
✔ Fallback graphics  

---

## 🔮 Future Improvements

- Add more emotions (fear, disgust, neutral filters)
- Implement video recording functionality
- GPU acceleration for higher FPS
- Mobile app version
- Custom filter creation interface
- Intensity-based effects (emotion confidence → effect strength)

---

## 👥 Contributors

Raghav Gupta [102215011]
Daksh Gautam [102215098]
Himanshu Jhawar [102215251]
Divyam Gupta [102215334]

---

## 📝 License

MIT License - Free to use and modify
