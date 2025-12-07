import cv2
import numpy as np
from fer import FER
import random

# Initialize the FER detector
detector = FER()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define colors for each emotion
emotion_colors = {
    'angry': (0, 0, 255),    # Red
    'disgust': (0, 255, 0),  # Green
    'fear': (128, 0, 128),   # Purple
    'happy': (0, 255, 255),  # Yellow
    'sad': (255, 0, 0),      # Blue
    'neutral': (128, 128, 128)  # Gray
}

# Load cloud image
cloud_img = cv2.imread(r"cloud.png", cv2.IMREAD_UNCHANGED)
cloud_img = cv2.resize(cloud_img, (100, 100))

# Function to overlay cloud with transparency
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect emotions
    results = detector.detect_emotions(frame)

    if results:
        for face in results:
            x, y, w, h = face['box']
            emotions = face['emotions']
            dominant_emotion = max(emotions, key=emotions.get)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), emotion_colors[dominant_emotion], 2)

            # Display emotion text
            cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, emotion_colors[dominant_emotion], 2)

            # Apply surprise effect (cloud with lightning)
            if dominant_emotion == 'surprise':
                cloud_x = x + w//2 - 50
                cloud_y = y - 120
                overlay_image_alpha(frame, cloud_img[:, :, :3], cloud_x, cloud_y, cloud_img[:, :, 3] / 255.0)

                # Add lightning effect
                if random.random() < 0.3:  # 30% chance of lightning
                    lightning_start = (cloud_x + random.randint(20, 80), cloud_y + 80)
                    lightning_end = (lightning_start[0] + random.randint(-20, 20), lightning_start[1] + random.randint(40, 60))
                    cv2.line(frame, lightning_start, lightning_end, (255, 255, 255), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
