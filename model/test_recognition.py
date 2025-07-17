import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow info/warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import cv2
from face_recognizer import FaceRecognizer
from database_manager import FaceDatabase

def main():
    # Initialize components
    recognizer = FaceRecognizer()
    db = FaceDatabase()
    
    # Camera setup with Windows backend
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("Face recognition running. Press 'x' to exit...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Process frame
        temp_path = "temp_cam.jpg"
        cv2.imwrite(temp_path, frame)
        
        try:
            # Recognize face
            name, confidence = recognizer.recognize_face(temp_path)
            os.remove(temp_path)
            
            # Display results
            label = f"{name} ({confidence:.2f})" if confidence > 0.6 else "Unknown"
            color = (0, 255, 0) if confidence > 0.6 else (0, 0, 255)
            cv2.putText(frame, label, (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Background logging (1-hour cooldown)
            if confidence > 0.7 and db.should_save_face(name):
                db.log_face(name, confidence)
                
        except Exception as e:
            print(f"Processing error: {str(e)}")
        
        cv2.imshow('Face Recognition - Press X to exit', frame)
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()