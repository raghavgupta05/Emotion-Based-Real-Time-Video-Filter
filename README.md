🎭 Emotion-Based Video Filter
Real-time emotion detection system that applies dynamic visual filters to webcam feed based on facial expressions.
Features

😊 Happy: Sunshine effect with yellow overlay
😢 Sad: Rain animation with blue tint
😮 Surprise: Storm clouds with lightning
😠 Angry: Red tint with rising fumes

Requirements
bashpip install opencv-python numpy fer tensorflow
Python 3.7+ | Webcam Required
Quick Start
bash# Clone and install
git clone https://github.com/yourusername/emotion-video-filter.git
cd emotion-video-filter
pip install -r requirements.txt

# Run
python final.py

# Press 'q' to quit
Optional Assets
Place these PNG images in the project folder for enhanced visuals:

sun.png - Sunshine graphic
cloud.png - Cloud graphic
fumes.png - Steam/fumes graphic

System uses fallback graphics if not provided.
How It Works

Captures webcam feed
Detects faces (Haar Cascade)
Analyzes emotions (FER + MTCNN)
Applies corresponding filter
Smooths transitions over 10 frames

Files

final.py - Main application (4 emotions)
V3.py - 3 emotions version
V1.py - Basic version (happy/sad only)

Troubleshooting
Low FPS? Lower webcam resolution or use GPU acceleration
No face detected? Ensure good lighting and face camera directly
Camera error? Check permissions and close other camera apps
