import cv2
import numpy as np
from fer import FER
from collections import deque
import random

# Global variables
sun_image = None
cloud_image = None
last_emotions = deque(maxlen=10)  # For smoothing

def load_images():
    global sun_image, cloud_image
    sun_image = cv2.imread('sun.png', cv2.IMREAD_UNCHANGED)
    if sun_image is None:
        print("Warning: sun.png not found. Using default sun.")
    cloud_image = cv2.imread('cloud.png', cv2.IMREAD_UNCHANGED)
    if cloud_image is None:
        print("Warning: cloud.png not found. Using default cloud.")

def apply_sunshine_filter(frame):
    global sun_image
    
    # Create a yellow overlay
    overlay = np.full(frame.shape, (0, 255, 255), dtype=np.uint8)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    if sun_image is not None:
        # Resize sun image to fit in the corner
        sun_size = min(frame.shape[0], frame.shape[1]) // 4
        resized_sun = cv2.resize(sun_image, (sun_size, sun_size))
        
        # Calculate position (top-left corner)
        y_offset = 10
        x_offset = 10
        
        # Get the alpha channel of the sun image
        if resized_sun.shape[2] == 4:
            alpha_sun = resized_sun[:, :, 3] / 255.0
            sun_rgb = resized_sun[:, :, :3]
        else:
            alpha_sun = np.ones((sun_size, sun_size))
            sun_rgb = resized_sun
        
        # Blend the sun image with the frame
        for c in range(0, 3):
            frame[y_offset:y_offset+sun_size, x_offset:x_offset+sun_size, c] = \
                (alpha_sun * sun_rgb[:, :, c] + 
                 (1 - alpha_sun) * frame[y_offset:y_offset+sun_size, x_offset:x_offset+sun_size, c])
    else:
        # Fallback to drawing a simple sun
        cv2.circle(frame, (50, 50), 30, (0, 255, 255), -1)
        cv2.circle(frame, (50, 50), 20, (255, 255, 0), -1)
    
    return frame

class RainDrop:
    def __init__(self, x, y, speed, length):
        self.x = x
        self.y = y
        self.speed = speed
        self.length = length

    def fall(self, height):
        self.y += self.speed
        if self.y > height:
            self.y = 0

def apply_rain_filter(frame, rain_drops):
    # Create a blue tint
    blue_tint = np.full(frame.shape, (130, 0, 0), dtype=np.uint8)
    cv2.addWeighted(blue_tint, 0.2, frame, 0.8, 0, frame)
    
    # Draw rain drops
    for drop in rain_drops:
        drop.fall(frame.shape[0])
        cv2.line(frame, (drop.x, drop.y), (drop.x, drop.y + drop.length), (200, 200, 200), 1)
    
    return frame

def apply_surprise_filter(frame, faces):
    global cloud_image
    
    for (x, y, w, h) in faces:
        if cloud_image is not None:
            # Resize cloud image
            cloud_size = min(frame.shape[0], frame.shape[1]) // 2
            resized_cloud = cv2.resize(cloud_image, (cloud_size, cloud_size))
            
            # Calculate position (above the face)
            cloud_y = max(0, y - cloud_size)
            cloud_x = x + w//2 - cloud_size//2
            
            # Get the alpha channel of the cloud image
            if resized_cloud.shape[2] == 4:
                alpha_cloud = resized_cloud[:, :, 3] / 255.0
                cloud_rgb = resized_cloud[:, :, :3]
            else:
                alpha_cloud = np.ones((cloud_size, cloud_size))
                cloud_rgb = resized_cloud
            
            # Blend the cloud image with the frame
            for c in range(0, 3):
                frame[cloud_y:cloud_y+cloud_size, cloud_x:cloud_x+cloud_size, c] = \
                    (alpha_cloud * cloud_rgb[:, :, c] + 
                     (1 - alpha_cloud) * frame[cloud_y:cloud_y+cloud_size, cloud_x:cloud_x+cloud_size, c])
        else:
            # Fallback to drawing a simple cloud
            cv2.ellipse(frame, (x+w//2, y-30), (w//3, 20), 0, 0, 360, (200, 200, 200), -1)

        # Add lightning effect
        if random.random() < 0.3:  # 30% chance of lightning
            lightning_start = (x + w//2, y - 30)
            lightning_end = (lightning_start[0] + random.randint(-w//4, w//4), y + h//2)
            cv2.line(frame, lightning_start, lightning_end, (255, 255, 255), 2)
    
    return frame

def initialize_rain(num_drops, width, height):
    return [RainDrop(np.random.randint(0, width), 
                     np.random.randint(0, height), 
                     np.random.randint(5, 15),
                     np.random.randint(5, 15)) 
            for _ in range(num_drops)]

def get_smoothed_emotion(emotion):
    global last_emotions
    last_emotions.append(emotion)
    return max(set(last_emotions), key=last_emotions.count)

def apply_filter(frame, emotion, rain_drops, faces):
    smoothed_emotion = get_smoothed_emotion(emotion)
    if smoothed_emotion == 'happy':
        return apply_sunshine_filter(frame)
    elif smoothed_emotion == 'sad':
        return apply_rain_filter(frame, rain_drops)
    elif smoothed_emotion == 'surprise':
        return apply_surprise_filter(frame, faces)
    else:
        return frame  # No filter for other emotions

def main():
    load_images()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(0)
    
    # Initialize rain drops
    rain_drops = initialize_rain(200, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print("Press 'q' to quit the application.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                
                try:
                    # Detect emotions
                    emotions = emotion_detector.detect_emotions(face)
                    
                    if emotions:
                        # Get the most probable emotion
                        emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                        
                        # Apply filter based on emotion
                        frame = apply_filter(frame, emotion, rain_drops, faces)
                        
                        # Draw rectangle and emotion text
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    else:
                        cv2.putText(frame, "No emotion detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                except Exception as e:
                    print(f"Error in emotion detection: {str(e)}")
                    cv2.putText(frame, "Error in emotion detection", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Emotion-based Filter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting application...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
