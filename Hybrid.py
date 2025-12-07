import cv2
import numpy as np
from fer import FER

# Global variable for loaded sun image
sun_image = cv2.imread(r"C:\Users\Abhimanyu kumar\OneDrive\Desktop\EDGE AI\UCS547- Accelerated Data Science\Project 1- IP\sun.png")

def load_sun_image():
    global sun_image
    sun_image = cv2.imread(r"C:\Users\Abhimanyu kumar\OneDrive\Desktop\EDGE AI\UCS547- Accelerated Data Science\Project 1- IP\sun.png", cv2.IMREAD_UNCHANGED)
    if sun_image is None:
        print("Warning: sun.png not found. Using default sun.")

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

def initialize_rain(num_drops, width, height):
    return [RainDrop(np.random.randint(0, width), 
                     np.random.randint(0, height), 
                     np.random.randint(5, 15),
                     np.random.randint(5, 15)) 
            for _ in range(num_drops)]

def apply_filter(frame, emotion, rain_drops):
    if emotion == 'happy':
        return apply_sunshine_filter(frame)
    elif emotion == 'sad':
        return apply_rain_filter(frame, rain_drops)
    else:
        return frame  # No filter for other emotions

def main():
    load_sun_image()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    emotion_detector = FER(mtcnn=True)

    cap = cv2.VideoCapture(0)
    
    # Initialize rain drops
    rain_drops = initialize_rain(200, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            # Detect emotions
            emotions = emotion_detector.detect_emotions(face)
            
            if emotions:
                # Get the most probable emotion
                emotion = max(emotions[0]['emotions'], key=emotions[0]['emotions'].get)
                
                # Apply filter based on emotion
                frame = apply_filter(frame, emotion, rain_drops)
                
                # Draw rectangle and emotion text
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Emotion-based Filter', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
