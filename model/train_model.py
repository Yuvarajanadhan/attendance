import os
import pickle
import numpy as np
from deepface import DeepFace
import pandas as pd

def train(dataset_path="../dataset", output_path="embeddings_db.pkl"):
    """Train face recognition model"""
    # Verify dataset exists
    if not os.path.exists(dataset_path):
        print(f"\nERROR: Dataset folder not found at {dataset_path}")
        print("Please create this structure:")
        print("FACE_Model/")
        print("├── model/")
        print("└── dataset/")
        print("    ├── Person1/")
        print("    │   ├── img1.jpg")
        print("    │   └── img2.jpg")
        print("    └── Person2/")
        print("        ├── photo1.jpg")
        print("        └── ...")
        return

    embeddings = []
    names = []
    
    print("\nStarting training...")
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"\nProcessing {person_name}:")
        for img_file in os.listdir(person_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_dir, img_file)
                try:
                    print(f"  Analyzing {img_file}...", end=" ")
                    embedding = DeepFace.represent(
                        img_path=img_path,
                        model_name="Facenet",
                        enforce_detection=True,
                        detector_backend="retinaface"
                    )[0]["embedding"]
                    embeddings.append(embedding)
                    names.append(person_name)
                    print("✓")
                except Exception as e:
                    print(f"✗ (Error: {str(e)})")
    
    if embeddings:
        df = pd.DataFrame({"name": names, "embedding": embeddings})
        df.to_pickle(output_path)
        print(f"\nTraining complete! Saved {len(names)} face embeddings to {output_path}")
        print(f"Recognized {len(set(names))} unique people.")
    else:
        print("\nTraining failed - no valid faces found")

if __name__ == "__main__":
    # Suppress TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    
    train()