import pickle
import numpy as np
from deepface import DeepFace
import cv2
import os

class FaceRecognizer:
    def __init__(self, model_path="embeddings_db.pkl"):
        self.embeddings, self.labels = self._load_model(model_path)
        
    def _load_model(self, path):
        """Load trained embeddings"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return np.array(data['embedding'].tolist()), data['name'].tolist()
    
    def recognize_face(self, image_path, threshold=0.6):
        """Recognize face from image file"""
        try:
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name="Facenet",
                enforce_detection=True,
                detector_backend="retinaface"
            )[0]["embedding"]
            
            input_vec = np.array(embedding)
            best_score = -1
            best_match = None
            
            for i, db_vec in enumerate(self.embeddings):
                similarity = np.dot(input_vec, db_vec) / (np.linalg.norm(input_vec) * np.linalg.norm(db_vec))
                if similarity > best_score:
                    best_score = similarity
                    best_match = i
            
            if best_score > threshold:
                return self.labels[best_match], float(best_score)
            return "Unknown", 0
            
        except Exception as e:
            print(f"Recognition error: {str(e)}")
            return "Error", 0

    def recognize_camera(self):
        """Real-time webcam recognition"""
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('Press Q to quit', frame)
            
            # Save temporary image
            temp_path = "temp_cam.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Recognize
            name, conf = self.recognize_face(temp_path)
            os.remove(temp_path)  # Clean up
            
            print(f"Detected: {name} ({conf:.2f})", end='\r')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()