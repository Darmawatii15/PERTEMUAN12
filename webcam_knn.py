import cv2
import torch
import joblib
import numpy as np
from utils_facenet import embed_from_image, align_mtcnn
from facenet_pytorch import MTCNN

# --- Load Model ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
knn = joblib.load("facenet_knn.joblib")
labels = knn.classes_

# --- Buka Webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka webcam")
    exit()

print("Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi wajah dengan MTCNN
    try:
        aligned = align_mtcnn(frame, mtcnn)
    except:
        aligned = None

    name = "Unknown"

    if aligned is not None:
        emb = embed_from_image(aligned)
        pred = knn.predict([emb])[0]
        prob = knn.predict_proba([emb])[0]
        best_prob = np.max(prob)

        if best_prob > 0.55:   # threshold pengenalan
            name = f"{pred} ({best_prob:.2f})"
        else:
            name = "Unknown"

        # Tampilkan bounding box
        box, _ = mtcnn.detect(frame)
        if box is not None:
            (x1, y1, x2, y2) = box[0].astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("FaceNet + KNN Realtime", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
